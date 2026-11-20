/* MATLAB gateway for the same native arf/acf core; no Python APIs or runtime.
 * Internal commands are wrapped by fastrank1pt_arf / arf_cbc_plan.
 */
#include "mex.h"
#include <math.h>
#include <stdbool.h>
#ifdef I
#undef I
#endif
#include "acf_point_native.c"

typedef struct Entry {uint64_t id;FFT *fft;CBC *cbc;double kernel_seconds,prepare_seconds;
    double fft_plan_seconds,kernel_evaluation_seconds,cbc_input_seconds;
    const char *kernel_method;
    struct Entry *next;} Entry;
static Entry *entries=NULL;
static uint64_t next_id=1;
static int registered=0;
static double (*native_get_d)(const R*,int);
static void (*native_bsetd)(Ball*,double),(*native_bseti)(Ball*,I);
static void (*native_bdivui)(Ball*,const Ball*,uint64_t,I);
static void (*native_bscale)(Ball*,const Ball*,I),(*native_bzeta)(Ball*,const Ball*,I);
static void (*native_exp_pi_i)(CBall*,const CBall*,I);
static void (*native_polylog)(CBall*,const CBall*,const CBall*,I);
static int (*native_bfinite)(const Ball*);
static I (*native_brelerror)(const Ball*);
static void (*native_cleanup)(void);
static void (*native_pi)(Ball*,I),(*native_bernoulli)(Ball*,uint64_t,I);
static void (*native_bpowui)(Ball*,const Ball*,uint64_t,I),(*native_factorial)(Ball*,uint64_t,I);
static void (*native_bdiv)(Ball*,const Ball*,const Ball*,I);

static void fail(const char *message){mexErrMsgIdAndTxt("arf_cbc:Native","%s",message);}
static void dispose(Entry *e){point_cbc_clear(e->cbc);point_plan_clear(e->fft);free(e);}
static void cleanup(void){while(entries){Entry *e=entries;entries=e->next;dispose(e);}point_shutdown();}
static double number(const mxArray *a,const char *name){
    if(!mxIsDouble(a)||mxIsComplex(a)||mxGetNumberOfElements(a)!=1||!isfinite(mxGetScalar(a)))
        mexErrMsgIdAndTxt("arf_cbc:Input","%s must be a finite real double scalar.",name);
    return mxGetScalar(a);
}
static const mxArray *field(const mxArray *a,const char *name){
    if(!mxIsStruct(a)||mxGetNumberOfElements(a)!=1)fail("Invalid source structure.");
    const mxArray *v=mxGetField(a,0,name);if(!v)mexErrMsgIdAndTxt("arf_cbc:Input","Missing field %s.",name);
    return v;
}
static void load_runtime(const mxArray *path){
    if(!mxIsChar(path))fail("FLINT path must be a character vector.");
    char *s=mxArrayToString(path);int ok=point_init(s);mxFree(s);if(!ok)fail(point_error());
    if(!registered){mexAtExit(cleanup);registered=1;LARGE_INTEGER tick;QueryPerformanceCounter(&tick);next_id=(uint64_t)tick.QuadPart;}
    FARPROC address=GetProcAddress(point_flint_module,"arf_get_d");
    memcpy(&native_get_d,&address,sizeof(native_get_d));if(!native_get_d)fail("Missing arf_get_d.");
}
/* LOAD returns int in the core. Keep gateway symbol resolution separate. */
#define NATIVE_LOAD(var,name) do{FARPROC address=GetProcAddress(point_flint_module,name);memcpy(&(var),&address,sizeof(var));if(!(var))fail("Missing native kernel primitive " name);}while(0)
static void load_kernel_symbols(void){
    NATIVE_LOAD(native_bsetd,"arb_set_d");NATIVE_LOAD(native_bseti,"arb_set_si");
    NATIVE_LOAD(native_bdivui,"arb_div_ui");NATIVE_LOAD(native_bscale,"arb_mul_2exp_si");
    NATIVE_LOAD(native_bzeta,"arb_zeta");NATIVE_LOAD(native_exp_pi_i,"acb_exp_pi_i");
    NATIVE_LOAD(native_polylog,"acb_polylog");NATIVE_LOAD(native_bfinite,"arb_is_finite");
    NATIVE_LOAD(native_cleanup,"flint_cleanup");
    NATIVE_LOAD(native_pi,"arb_const_pi");NATIVE_LOAD(native_bernoulli,"arb_bernoulli_ui");
    NATIVE_LOAD(native_bpowui,"arb_pow_ui");NATIVE_LOAD(native_factorial,"arb_fac_ui");
    NATIVE_LOAD(native_bdiv,"arb_div");
    NATIVE_LOAD(native_brelerror,"arb_rel_error_bits");
}
/* Bound the polynomial fast path; larger orders retain the generic formula.
 * Exact integer detection never changes a noninteger smoothness parameter. */
static I bernoulli_order(double alpha){return alpha>=1 && alpha<=64 && alpha==floor(alpha)?(I)alpha:0;}
static const char *kernel_method(double alpha){return bernoulli_order(alpha)?"bernoulli-horner":"polylog";}
static void eval_bernoulli(R *out,const I *perm,I n,const double *x,I length,I order,I bits,int threads){
    const I degree=2*order,precision=bits+80+degree;
    R *coefficients=ralloc(degree+1);R choose={0},constant={0},integer={0};
    Ball pi={0},power={0},factorial={0},bernoulli={0};
    /* omega(x)=(-1)^(order+1)*(2*pi)^(2*order)/(2*order)! * B_(2*order)(x).
     * All constants/coefficient arithmetic and grid division stay high precision.
     * Guarded preparation uses Arb for constants only; the bulk Horner loop is arf. */
    native_pi(&pi,precision);native_bscale(&pi,&pi,1);
    native_bpowui(&power,&pi,(uint64_t)degree,precision);
    native_factorial(&factorial,(uint64_t)degree,precision);
    native_bdiv(&power,&power,&factorial,precision);rset(&constant,&power.mid);
    if(order%2==0)rneg(&constant,&constant);
    rseti(&choose,1);
    for(I k=0;k<=degree;k++){
        native_bernoulli(&bernoulli,(uint64_t)k,precision);
        rmul(coefficients+k,&bernoulli.mid,&choose,precision,RN);
        rmul(coefficients+k,coefficients+k,&constant,precision,RN);
        if(k<degree){rseti(&integer,degree-k);rmul(&choose,&choose,&integer,precision,RN);
            rseti(&integer,k+1);rdiv(&choose,&choose,&integer,precision,RN);}
    }
    bclear(&pi);bclear(&power);bclear(&factorial);bclear(&bernoulli);
    rclear(&choose);rclear(&constant);rclear(&integer);
    #pragma omp parallel num_threads(threads)
    {
        R grid={0},value={0},denominator={0},whole={0};rseti(&denominator,n);
        #pragma omp for schedule(static)
        for(I j=0;j<length;j++){
            if(perm){rseti(&grid,j?perm[j-1]:0);rdiv(&grid,&grid,&denominator,precision,RN);}
            else {
                /* Preserve the periodic extension, including negative inputs.
                 * Subtraction is high precision, not a double fractional part. */
                rsetd(&grid,x[j]);rsetd(&whole,floor(x[j]));rsub(&grid,&grid,&whole,precision,RN);
            }
            rset(&value,coefficients);
            for(I k=1;k<=degree;k++){
                rmul(&value,&value,&grid,precision,RN);radd(&value,&value,coefficients+k,precision,RN);
            }
            rround(out+j,&value,bits,RN);
        }
        rclear(&grid);rclear(&value);rclear(&denominator);rclear(&whole);native_cleanup();
    }
    rfree(coefficients,degree+1);
}
static void eval_polylog(R *out,const I *perm,I n,const double *x,I length,double alpha,I bits,int threads,I minimum_accuracy){
    int invalid=0;I precision=bits+80;
    #pragma omp parallel num_threads(threads)
    {
        CBall order={0},argument={0},unit={0},value={0};
        native_bsetd(&order.re,2*alpha);
        #pragma omp for schedule(static) reduction(|:invalid)
        for(I j=0;j<length;j++){
            I work=precision;
            for(;;){
                if((perm && j==0)||(!perm && x[j]==0))native_bzeta(&value.re,&order.re,work);
                else {
                    if(perm){native_bseti(&argument.re,perm[j-1]);native_bdivui(&argument.re,&argument.re,(uint64_t)n,work);}
                    else native_bsetd(&argument.re,x[j]);
                    native_bscale(&argument.re,&argument.re,1);
                    native_exp_pi_i(&unit,&argument,work);native_polylog(&value,&order,&unit,work);
                }
                /* Only the private reference command adapts its guard precision.
                 * Production noninteger evaluation keeps the existing bits+80 path. */
                if(!minimum_accuracy || (native_bfinite(&value.re) && native_brelerror(&value.re)<=-minimum_accuracy))break;
                if(work>=16384){invalid=1;break;}work*=2;
            }
            if(!native_bfinite(&value.re))invalid=1;
            rround(out+j,&value.re.mid,bits,RN);rscale(out+j,out+j,1);
        }
        bclear(&order.re);bclear(&order.im);bclear(&argument.re);bclear(&argument.im);
        bclear(&unit.re);bclear(&unit.im);bclear(&value.re);bclear(&value.im);
        native_cleanup();
    }
    if(invalid)fail("Native kernel evaluation returned a nonfinite value.");
}
static void eval_kernel(R *out,const I *perm,I n,const double *x,I length,double alpha,I bits,int threads){
    I order=bernoulli_order(alpha);
    if(order)eval_bernoulli(out,perm,n,x,length,order,bits,threads);
    else eval_polylog(out,perm,n,x,length,alpha,bits,threads,0);
}
static Entry *lookup(const mxArray *token){
    if(!mxIsUint64(token)||mxGetNumberOfElements(token)!=1)fail("Invalid plan token.");
    uint64_t id=*(uint64_t*)mxGetData(token);
    for(Entry *e=entries;e;e=e->next)if(e->id==id)return e;
    mexErrMsgIdAndTxt("arf_cbc:Handle","Plan is invalid or has already been released.");return NULL;
}
static mxArray *strings(const Z *v,I d){
    mxArray *out=mxCreateCellMatrix((mwSize)d,1);
    for(I j=0;j<d;j++){char *s=point_string(v,j,0,80);mxSetCell(out,(mwIndex)j,mxCreateString(s));point_string_free(s);}
    return out;
}
static mxArray *prepare_info(const Entry *e){
    const char *fields[]={"bits","threads","fft_length","kernel_seconds","preparation_seconds","python_required",
        "fft_plan_seconds","kernel_evaluation_seconds","cbc_input_seconds","kernel_method"};
    mxArray *s=mxCreateStructMatrix(1,1,10,fields);
    mxSetField(s,0,"bits",mxCreateDoubleScalar((double)e->fft->bits));
    mxSetField(s,0,"threads",mxCreateDoubleScalar(e->fft->threads));
    mxSetField(s,0,"fft_length",mxCreateDoubleScalar((double)e->fft->n));
    mxSetField(s,0,"kernel_seconds",mxCreateDoubleScalar(e->kernel_seconds));
    mxSetField(s,0,"preparation_seconds",mxCreateDoubleScalar(e->prepare_seconds));
    mxSetField(s,0,"python_required",mxCreateLogicalScalar(false));
    mxSetField(s,0,"fft_plan_seconds",mxCreateDoubleScalar(e->fft_plan_seconds));
    mxSetField(s,0,"kernel_evaluation_seconds",mxCreateDoubleScalar(e->kernel_evaluation_seconds));
    mxSetField(s,0,"cbc_input_seconds",mxCreateDoubleScalar(e->cbc_input_seconds));
    mxSetField(s,0,"kernel_method",mxCreateString(e->kernel_method));return s;
}
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
    if(nrhs<1||!mxIsChar(prhs[0]))fail("A command is required.");
    char command[32];mxGetString(prhs[0],command,sizeof(command));
    if(!strcmp(command,"prepare")){
        if(nrhs!=10||nlhs<1||nlhs>2)fail("Invalid prepare arguments.");
        double nd=number(prhs[2],"N"),dd=number(prhs[3],"dimension"),bd=number(prhs[4],"bits"),td=number(prhs[5],"threads");
        if(nd<3||nd!=floor(nd)||fmod(nd,2)!=1||nd*nd>=9007199254740992.0||dd<1||dd!=floor(dd)||bd<32||bd>512||bd!=floor(bd)||td<1||td>64||td!=floor(td))fail("Invalid N, dimension, bits or threads.");
        I n=(I)nd,d=(I)dd,m=(n-1)/2,L=2;while(L<2*m-1)L<<=1;
        if(!mxIsInt64(prhs[6])||mxGetNumberOfElements(prhs[6])!=(mwSize)m)fail("Expected int64 permutation.");
        for(int a=7;a<=8;a++){
            if(!mxIsDouble(prhs[a])||mxIsComplex(prhs[a])||mxGetNumberOfElements(prhs[a])!=(mwSize)d)fail("Invalid weight arrays.");
            for(I j=0;j<d;j++)if(!isfinite(mxGetPr(prhs[a])[j])||mxGetPr(prhs[a])[j]<=0)fail("Weights must be finite and positive.");
        }
        I *perm=(I*)mxGetData(prhs[6]);unsigned char *seen=calloc((size_t)m+1,1);
        for(I j=0;j<m;j++)if(perm[j]<1||perm[j]>m||seen[perm[j]]){free(seen);fail("Invalid permutation.");}else seen[perm[j]]=1;
        free(seen);load_runtime(prhs[1]);load_kernel_symbols();
        double clock=seconds();FFT *fft=point_plan(L,(I)bd,(int)td);if(!fft)fail("Cannot allocate FFT plan.");
        double fft_plan_seconds=seconds()-clock;
        const mxArray *source=prhs[9];const mxArray *kind_value=field(source,"kind");
        if(!mxIsChar(kind_value)){point_plan_clear(fft);fail("Invalid kernel source kind.");}
        char *kind=mxArrayToString(kind_value);
        CBC *cbc=NULL;double kernel_clock=seconds(),kernel_evaluation_seconds=0,cbc_input_seconds=0;
        const char *method="supplied-values";
        if(!strcmp(kind,"cache")){
            const mxArray *filename=field(source,"filename");if(!mxIsChar(filename)){point_plan_clear(fft);mxFree(kind);fail("Invalid kernel cache filename.");}
            char *path=mxArrayToString(filename);
            cbc=point_cbc_prepare(fft,n,d,perm,mxGetPr(prhs[7]),path);mxFree(path);
            method="cached-values";
        }else if(!strcmp(kind,"native")){
            double alpha=number(field(source,"alpha"),"alpha");if(alpha<=0.5){point_plan_clear(fft);mxFree(kind);fail("Require alpha>1/2.");}
            R *values=ralloc(m+1);eval_kernel(values,perm,n,NULL,m+1,alpha,(I)bd,(int)td);
            kernel_evaluation_seconds=seconds()-kernel_clock;double input_clock=seconds();
            cbc=point_cbc_prepare_values(fft,n,d,perm,mxGetPr(prhs[7]),values);rfree(values,m+1);
            cbc_input_seconds=seconds()-input_clock;method=kernel_method(alpha);
        }else if(!strcmp(kind,"values")){
            const mxArray *values=field(source,"values");
            if(mxGetNumberOfElements(values)!=(mwSize)(m+1)||(!mxIsDouble(values)&&!mxIsCell(values))||mxIsComplex(values)){point_plan_clear(fft);mxFree(kind);fail("Invalid kernel values.");}
            R *v=ralloc(m+1);Ball tmp={0};int invalid=0;
            for(I j=0;j<m+1;j++){
                if(mxIsDouble(values)){double x=mxGetPr(values)[j];if(!isfinite(x)){invalid=1;break;}rsetd(v+j,x);rround(v+j,v+j,(I)bd,RN);}
                else {const mxArray *cell=mxGetCell(values,(mwIndex)j);if(!cell||!mxIsChar(cell)){invalid=1;break;}
                    char *s=mxArrayToString(cell);int bad=bparse(&tmp,s,(I)bd+80);mxFree(s);if(bad){invalid=1;break;}rround(v+j,&tmp.mid,(I)bd,RN);}
            }
            if(!invalid)cbc=point_cbc_prepare_values(fft,n,d,perm,mxGetPr(prhs[7]),v);
            bclear(&tmp);rfree(v,m+1);
        }
        mxFree(kind);if(!cbc){point_plan_clear(fft);fail("Could not prepare the kernel inputs.");}
        point_set_beta(cbc,mxGetPr(prhs[8]));
        Entry *e=calloc(1,sizeof(Entry));e->id=next_id++;e->fft=fft;e->cbc=cbc;
        e->kernel_seconds=seconds()-kernel_clock;e->prepare_seconds=seconds()-clock;e->next=entries;entries=e;mexLock();
        e->fft_plan_seconds=fft_plan_seconds;e->kernel_evaluation_seconds=kernel_evaluation_seconds;
        e->cbc_input_seconds=cbc_input_seconds;e->kernel_method=method;
        plhs[0]=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);*(uint64_t*)mxGetData(plhs[0])=e->id;
        if(nlhs>1)plhs[1]=prepare_info(e);
    }else if(!strcmp(command,"run")){
        if(nrhs!=3||nlhs<2||nlhs>3)fail("Invalid run arguments.");
        Entry *e=lookup(prhs[1]);I d=e->cbc->d;int audit=mxIsLogicalScalarTrue(prhs[2]);
        double started=seconds();I *g=malloc((size_t)d*sizeof(I));double *stages=calloc((size_t)d,sizeof(double));
        Z *Pval=point_vec(d),*updated=point_vec(d),*gaps=point_vec(d);
        double core_clock=seconds();point_cbc(e->cbc,g,Pval,updated,gaps,stages,audit);double core_seconds=seconds()-core_clock;
        plhs[0]=mxCreateDoubleMatrix((mwSize)d,1,mxREAL);plhs[1]=mxCreateDoubleMatrix((mwSize)d,1,mxREAL);
        for(I j=0;j<d;j++){mxGetPr(plhs[0])[j]=(double)g[j];mxGetPr(plhs[1])[j]=native_get_d(&Pval[j].re,RN);}
        if(nlhs>2){
            const char *fields[]={"P","updated_P","candidate_gaps","stage_seconds","cbc_seconds","mex_seconds","bits","threads","fft_length","interval_certified","python_required","kernel_preparation_reused"};
            mxArray *s=mxCreateStructMatrix(1,1,12,fields);
            mxSetField(s,0,"P",strings(Pval,d));mxSetField(s,0,"updated_P",strings(updated,d));mxSetField(s,0,"candidate_gaps",strings(gaps,d));
            mxArray *t=mxCreateDoubleMatrix((mwSize)d,1,mxREAL);memcpy(mxGetPr(t),stages,(size_t)d*sizeof(double));mxSetField(s,0,"stage_seconds",t);
            mxSetField(s,0,"cbc_seconds",mxCreateDoubleScalar(core_seconds));
            mxSetField(s,0,"mex_seconds",mxCreateDoubleScalar(seconds()-started));
            mxSetField(s,0,"bits",mxCreateDoubleScalar((double)e->fft->bits));mxSetField(s,0,"threads",mxCreateDoubleScalar(e->fft->threads));
            mxSetField(s,0,"fft_length",mxCreateDoubleScalar((double)e->fft->n));mxSetField(s,0,"interval_certified",mxCreateLogicalScalar(false));
            mxSetField(s,0,"python_required",mxCreateLogicalScalar(false));mxSetField(s,0,"kernel_preparation_reused",mxCreateLogicalScalar(true));plhs[2]=s;
        }
        free(g);free(stages);point_vec_clear(Pval,d);point_vec_clear(updated,d);point_vec_clear(gaps,d);
    }else if(!strcmp(command,"free")){
        if(nrhs!=2||nlhs>1||!mxIsUint64(prhs[1])||mxGetNumberOfElements(prhs[1])!=1)fail("Invalid free arguments.");
        uint64_t id=*(uint64_t*)mxGetData(prhs[1]);Entry **cursor=&entries;int found=0;
        while(*cursor){if((*cursor)->id==id){Entry *e=*cursor;*cursor=e->next;dispose(e);mexUnlock();found=1;break;}cursor=&(*cursor)->next;}
        if(nlhs)plhs[0]=mxCreateLogicalScalar(found);
    }else if(!strcmp(command,"status")){
        if(nrhs!=1||nlhs>1)fail("Invalid status arguments.");
        const char *fields[]={"active_plans","runtime_loaded","runtime_path","python_runtime_loaded","advanpix_loaded"};
        mxArray *s=mxCreateStructMatrix(1,1,5,fields);int count=0;for(Entry *e=entries;e;e=e->next)count++;
        char path[4096]="";if(point_flint_module)GetModuleFileNameA(point_flint_module,path,sizeof(path));
        mxSetField(s,0,"active_plans",mxCreateDoubleScalar(count));mxSetField(s,0,"runtime_loaded",mxCreateLogicalScalar(point_flint_module!=NULL));
        mxSetField(s,0,"runtime_path",mxCreateString(path));mxSetField(s,0,"python_runtime_loaded",mxCreateLogicalScalar(GetModuleHandleA("python311.dll")!=NULL));
        mxSetField(s,0,"advanpix_loaded",mxCreateLogicalScalar(GetModuleHandleA("mpimpl.mexw64")!=NULL));
        if(nlhs)plhs[0]=s;else mxDestroyArray(s);
    }else if(!strcmp(command,"kernel")||!strcmp(command,"kernel_polylog")){
        if(nrhs!=6||nlhs<1||nlhs>2)fail("Invalid kernel arguments.");
        load_runtime(prhs[1]);load_kernel_symbols();
        double alpha=number(prhs[2],"alpha"),bits=number(prhs[4],"bits"),threads=number(prhs[5],"threads");
        const mxArray *x=prhs[3];if(alpha<=0.5||bits<32||bits>512||bits!=floor(bits)||threads<1||threads>64||threads!=floor(threads)||!mxIsDouble(x)||mxIsComplex(x))fail("Invalid kernel arguments.");
        I length=(I)mxGetNumberOfElements(x);for(I j=0;j<length;j++)if(!isfinite(mxGetPr(x)[j]))fail("Nonfinite kernel grid.");
        int force_polylog=!strcmp(command,"kernel_polylog");R *v=ralloc(length);
        if(force_polylog)eval_polylog(v,NULL,0,mxGetPr(x),length,alpha,(I)bits,(int)threads,(I)bits+32);
        else eval_kernel(v,NULL,0,mxGetPr(x),length,alpha,(I)bits,(int)threads);
        plhs[0]=mxCreateNumericArray(mxGetNumberOfDimensions(x),mxGetDimensions(x),mxDOUBLE_CLASS,mxREAL);
        for(I j=0;j<length;j++)mxGetPr(plhs[0])[j]=native_get_d(v+j,RN);
        if(nlhs>1){
            const char *fields[]={"values","kernel_method","bits"};mxArray *s=mxCreateStructMatrix(1,1,3,fields);
            mxArray *values=mxCreateCellMatrix((mwSize)length,1);
            for(I j=0;j<length;j++){char *value=rstr(v+j,80);mxSetCell(values,(mwIndex)j,mxCreateString(value));ffree(value);}
            mxSetField(s,0,"values",values);mxSetField(s,0,"kernel_method",mxCreateString(force_polylog?"polylog":kernel_method(alpha)));
            mxSetField(s,0,"bits",mxCreateDoubleScalar(bits));plhs[1]=s;
        }
        rfree(v,length);
    }else fail("Unknown command.");
}
