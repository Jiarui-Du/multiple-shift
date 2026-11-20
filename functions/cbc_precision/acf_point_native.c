/* Ordinary high-precision FFT and same-criterion CBC, FLINT 3.3.1 Win64.
 * Project implementation; arithmetic primitives are from FLINT (LGPL).
 * Layouts follow upstream arf_types.h/acf_types.h. Round-to-nearest-even.
 * Arb is used ONLY to import decimal input / prepare roots outside timings.
 * FFT butterflies, q, candidate comparisons and P use arf/acf without radii.
 */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
typedef int64_t I;
typedef struct { I exp,size; uint64_t mantissa[2]; } R;
typedef struct { R re,im; } Z;
typedef struct { I exp; uint64_t man; } Mag;
typedef struct { R mid; Mag rad; } Ball;
typedef struct { Ball re,im; } CBall;
_Static_assert(sizeof(R)==32 && sizeof(Z)==64 && sizeof(CBall)==96,"Win64 FLINT ABI");
#define API __declspec(dllexport)
#define RN 4
static char error_text[256];
static HMODULE point_flint_module=NULL;
static char point_flint_path[4096];
static void (*rset)(R*,const R*),(*rsetd)(R*,double),(*rseti)(R*,I),(*rclear)(R*);
static void (*rneg)(R*,const R*),(*rscale)(R*,const R*,I);
static int (*rround)(R*,const R*,I,int),(*radd)(R*,const R*,const R*,I,int);
static int (*rsub)(R*,const R*,const R*,I,int),(*rmul)(R*,const R*,const R*,I,int);
static int (*rdiv)(R*,const R*,const R*,I,int),(*rsum)(R*,const R*,I,I,int);
static int (*rcmp)(const R*,const R*);
static int (*zmul)(Z*,const Z*,const Z*,I,int),(*zadd)(Z*,const Z*,const Z*,I,int);
static int (*zsub)(Z*,const Z*,const Z*,I,int);
static void (*zclear)(Z*);
static R *(*ralloc)(I);static void (*rfree)(R*,I);
static CBall *(*ballalloc)(I);static void (*ballfree)(CBall*,I);
static void (*roots)(CBall*,I,I,I),(*bclear)(Ball*);
static int (*bparse)(Ball*,const char*,I);
static char *(*rstr)(const R*,I);static void (*ffree)(void*);
#define LOAD(var,name) do { FARPROC address=GetProcAddress(h,name);memcpy(&(var),&address,sizeof(var)); if(!(var)){snprintf(error_text,sizeof(error_text),"Missing %s",name);return 0;} } while(0)
API int point_init(const char *dll) {
    if(point_flint_module){
        if(strcmp(point_flint_path,dll)){strcpy(error_text,"A different FLINT runtime is already active");return 0;}
        return 1;
    }
    HMODULE h=LoadLibraryExA(dll,NULL,LOAD_WITH_ALTERED_SEARCH_PATH);
    if(!h){snprintf(error_text,sizeof(error_text),"Cannot load FLINT DLL (Windows error %lu)",(unsigned long)GetLastError());return 0;}
    LOAD(rset,"arf_set");LOAD(rsetd,"arf_set_d");LOAD(rseti,"arf_set_si");LOAD(rclear,"arf_clear");
    LOAD(rneg,"arf_neg");LOAD(rscale,"arf_mul_2exp_si");LOAD(rround,"arf_set_round");
    LOAD(radd,"arf_add");LOAD(rsub,"arf_sub");LOAD(rmul,"arf_mul_rnd_any");
    LOAD(rdiv,"arf_div");LOAD(rsum,"arf_sum");LOAD(rcmp,"arf_cmp");
    LOAD(zmul,"acf_mul");LOAD(zadd,"acf_add");LOAD(zsub,"acf_sub");LOAD(zclear,"acf_clear");
    LOAD(ralloc,"_arf_vec_init");LOAD(rfree,"_arf_vec_clear");
    LOAD(ballalloc,"_acb_vec_init");LOAD(ballfree,"_acb_vec_clear");
    LOAD(roots,"_acb_vec_unit_roots");LOAD(bparse,"arb_set_str");LOAD(bclear,"arb_clear");
    LOAD(rstr,"arf_get_str");LOAD(ffree,"flint_free");
    point_flint_module=h;snprintf(point_flint_path,sizeof(point_flint_path),"%s",dll);
    omp_set_dynamic(0);return 1;
}
API void point_shutdown(void){if(point_flint_module){
    void (*cleanup_runtime)(void)=NULL;FARPROC address=GetProcAddress(point_flint_module,"flint_cleanup");
    memcpy(&cleanup_runtime,&address,sizeof(cleanup_runtime));if(cleanup_runtime)cleanup_runtime();
    FreeLibrary(point_flint_module);point_flint_module=NULL;point_flint_path[0]=0;}}
API const char *point_error(void){return error_text;}
API void *point_vec(I n){return ralloc(2*n);}
API void point_vec_clear(Z *v,I n){if(v)rfree((R*)v,2*n);}
API void point_load(Z *v,const double *re,const double *im,I n,I bits){
    for(I j=0;j<n;j++) {rsetd(&v[j].re,re[j]);rround(&v[j].re,&v[j].re,bits,RN);
        rsetd(&v[j].im,im?im[j]:0);rround(&v[j].im,&v[j].im,bits,RN);}
}
API char *point_string(const Z *v,I j,int imaginary,I digits){return rstr(imaginary?&v[j].im:&v[j].re,digits);}
API void point_string_free(char *s){ffree(s);}
typedef struct FFT {I n,bits,root_stride;int threads,logn,reverse_shift,real_backend;
    Z *w,*packed,*halfout;uint32_t *reverse;struct FFT *half;} FFT;
static void zero_r(R *r);
API void *point_plan(I n,I bits,int threads){
    if(n<2 || (n&(n-1)) || n>UINT32_MAX || bits<32 || bits>512 || threads<1 || threads>64) return NULL;
    FFT *p=calloc(1,sizeof(FFT));if(!p)return NULL;
    p->n=n;p->bits=bits;p->threads=threads;p->root_stride=1;p->real_backend=1;
    for(I k=n;k>1;k>>=1)p->logn++;
    p->w=(Z*)ralloc(n);p->reverse=malloc(n*sizeof(uint32_t));
    if(!p->w || !p->reverse)abort();
    for(uint32_t j=0;j<(uint32_t)n;j++) {uint32_t a=j,b=0;for(int t=0;t<p->logn;t++){b=(b<<1)|(a&1);a>>=1;}p->reverse[j]=b;}
    /* Root preparation is excluded; no balls are retained in FFT plan. */
    CBall *b=ballalloc(n/2);roots(b,-n,n/2,bits+48);
    for(I j=0;j<n/2;j++){rround(&p->w[j].re,&b[j].re.mid,bits,RN);rround(&p->w[j].im,&b[j].im.mid,bits,RN);}
    ballfree(b,n/2);
    p->half=calloc(1,sizeof(FFT));p->half->n=n/2;p->half->bits=bits;p->half->threads=threads;
    p->half->logn=p->logn-1;p->half->w=p->w;p->half->reverse=p->reverse;
    p->half->root_stride=2;p->half->reverse_shift=1;
    p->packed=(Z*)ralloc(n);p->halfout=(Z*)ralloc(n);
    return p;
}
API void point_plan_clear(FFT *p){if(p){rfree((R*)p->w,p->n);free(p->reverse);
    rfree((R*)p->packed,p->n);rfree((R*)p->halfout,p->n);free(p->half);free(p);}}
API void point_real_mode(FFT *p,int enabled){p->real_backend=enabled!=0;}
API int point_fft(FFT *p,Z *out,const Z *in,int inverse){
    if(!p || out==in)return 0;
    const I n=p->n,bits=p->bits;
    #pragma omp parallel num_threads(p->threads)
    {
        Z t={0};
        #pragma omp for schedule(static)
        for(I j=0;j<n;j++){
            Z *v=out+(p->reverse[j]>>p->reverse_shift);rset(&v->re,&in[j].re);
            if(inverse)rneg(&v->im,&in[j].im);else rset(&v->im,&in[j].im);
        }
        for(I half=1;half<n;half<<=1){
            I mask=half-1,step=n/(2*half);
            #pragma omp for schedule(static)
            for(I j=0;j<n/2;j++){
                I a=j+(j&~mask),b=a+half,k=j&mask;
                if(k==0){rset(&t.re,&out[b].re);rset(&t.im,&out[b].im);}
                else zmul(&t,out+b,p->w+k*step*p->root_stride,bits,RN);
                zsub(out+b,out+a,&t,bits,RN);zadd(out+a,out+a,&t,bits,RN);
            }
        }
        if(inverse){
            #pragma omp for schedule(static)
            for(I j=0;j<n;j++){
                rscale(&out[j].re,&out[j].re,-p->logn);
                rscale(&out[j].im,&out[j].im,-p->logn);rneg(&out[j].im,&out[j].im);
            }
        }
        zclear(&t);
    }
    return 1;
}
API int point_rfft(FFT *p,Z *out,const Z *in){
    I M=p->n/2,bits=p->bits;
    #pragma omp parallel for num_threads(p->threads) schedule(static)
    for(I j=0;j<M;j++){rset(&p->packed[j].re,&in[2*j].re);rset(&p->packed[j].im,&in[2*j+1].re);}
    point_fft(p->half,p->halfout,p->packed,0);
    radd(&out[0].re,&p->halfout[0].re,&p->halfout[0].im,bits,RN);zero_r(&out[0].im);
    rsub(&out[M].re,&p->halfout[0].re,&p->halfout[0].im,bits,RN);zero_r(&out[M].im);
    #pragma omp parallel num_threads(p->threads)
    {
        Z B={0},S={0},D={0},T={0};
        #pragma omp for schedule(static)
        for(I k=1;k<M;k++){
            rset(&B.re,&p->halfout[M-k].re);rneg(&B.im,&p->halfout[M-k].im);
            zadd(&S,p->halfout+k,&B,bits,RN);zsub(&D,p->halfout+k,&B,bits,RN);
            zmul(&T,&D,p->w+k,bits,RN);
            radd(&out[k].re,&S.re,&T.im,bits,RN);rscale(&out[k].re,&out[k].re,-1);
            rsub(&out[k].im,&S.im,&T.re,bits,RN);rscale(&out[k].im,&out[k].im,-1);
            rset(&out[p->n-k].re,&out[k].re);rneg(&out[p->n-k].im,&out[k].im);
        }
        zclear(&B);zclear(&S);zclear(&D);zclear(&T);
    }
    return 1;
}
API int point_irfft(FFT *p,Z *out,const Z *in){
    I M=p->n/2,bits=p->bits;
    radd(&p->packed[0].re,&in[0].re,&in[M].re,bits,RN);rscale(&p->packed[0].re,&p->packed[0].re,-1);
    rsub(&p->packed[0].im,&in[0].re,&in[M].re,bits,RN);rscale(&p->packed[0].im,&p->packed[0].im,-1);
    #pragma omp parallel num_threads(p->threads)
    {
        Z B={0},S={0},D={0},T={0},W={0};
        #pragma omp for schedule(static)
        for(I k=1;k<M;k++){
            rset(&B.re,&in[M-k].re);rneg(&B.im,&in[M-k].im);
            zadd(&S,in+k,&B,bits,RN);zsub(&D,in+k,&B,bits,RN);
            rset(&W.re,&p->w[k].re);rneg(&W.im,&p->w[k].im);zmul(&T,&D,&W,bits,RN);
            rsub(&p->packed[k].re,&S.re,&T.im,bits,RN);rscale(&p->packed[k].re,&p->packed[k].re,-1);
            radd(&p->packed[k].im,&S.im,&T.re,bits,RN);rscale(&p->packed[k].im,&p->packed[k].im,-1);
        }
        zclear(&B);zclear(&S);zclear(&D);zclear(&T);zclear(&W);
    }
    point_fft(p->half,p->halfout,p->packed,1);
    #pragma omp parallel for num_threads(p->threads) schedule(static)
    for(I j=0;j<M;j++){
        rset(&out[2*j].re,&p->halfout[j].re);rset(&out[2*j+1].re,&p->halfout[j].im);
        zero_r(&out[2*j].im);zero_r(&out[2*j+1].im);
    }
    return 1;
}
API void point_product(FFT *p,Z *out,const Z *a,const Z *b){
    #pragma omp parallel for num_threads(p->threads) schedule(static)
    for(I j=0;j<p->n;j++)zmul(out+j,a+j,b+j,p->bits,RN);
}
API void point_fold(FFT *p,Z *out,const Z *raw,I m){
    #pragma omp parallel for num_threads(p->threads) schedule(static)
    for(I j=0;j<m;j++){
        if(j<m-1)radd(&out[j].re,&raw[j].re,&raw[m+j].re,p->bits,RN);
        else rset(&out[j].re,&raw[j].re);
    }
}
API void point_maxdiff(FFT *p,Z *out,const Z *x,const Z *y){
    R delta={0},maximum={0},zero={0};
    for(I j=0;j<p->n;j++)for(int part=0;part<2;part++){
        rsub(&delta,part?&x[j].im:&x[j].re,part?&y[j].im:&y[j].re,p->bits+32,RN);
        if(rcmp(&delta,&zero)<0)rneg(&delta,&delta);
        if(rcmp(&delta,&maximum)>0)rset(&maximum,&delta);
    }
    rset(&out->re,&maximum);rclear(&delta);rclear(&maximum);
}
typedef struct {
    FFT *fft;I n,m,d;I *perm,*lookup;R *psi,*gamma,*beta,psi0,sumpsi;
    Z *F,*A,*input,*work,*product,*raw;R *q,*scores;
} CBC;
API void point_cbc_clear(CBC *c);
static double seconds(void){static LARGE_INTEGER f={0};LARGE_INTEGER x;if(!f.QuadPart)QueryPerformanceFrequency(&f);QueryPerformanceCounter(&x);return (double)x.QuadPart/f.QuadPart;}
static void zero_r(R *r){rclear(r);memset(r,0,sizeof(R));}
static CBC *point_cbc_allocate(FFT *fft,I n,I d,const I *perm,const double *gamma){
    I m=(n-1)/2,L=fft->n,bits=fft->bits;
    if(n<3 || d<1 || 2*m-1>L)return NULL;
    CBC *c=calloc(1,sizeof(CBC));if(!c)return NULL;
    c->fft=fft;c->n=n;c->m=m;c->d=d;
    c->perm=malloc(m*sizeof(I));memcpy(c->perm,perm,m*sizeof(I));
    c->lookup=calloc(n,sizeof(I));for(I j=0;j<m;j++)c->lookup[perm[j]]=c->lookup[n-perm[j]]=j;
    c->psi=ralloc(m);c->gamma=ralloc(d);c->beta=ralloc(d);c->q=ralloc(m);c->scores=ralloc(m);
    for(I j=0;j<d;j++){rsetd(c->gamma+j,gamma[j]);rseti(c->beta+j,1);}
    (void)bits;return c;
}
static CBC *point_cbc_finish(CBC *c){
    FFT *fft=c->fft;I L=fft->n,bits=fft->bits,m=c->m;
    rsum(&c->sumpsi,c->psi,m,bits,RN);
    c->F=point_vec(L);c->A=point_vec(L);c->input=point_vec(L);
    c->work=point_vec(L);c->product=point_vec(L);c->raw=point_vec(L);
    for(I j=0;j<m;j++)rset(&c->input[j].re,c->psi+j);
    if(fft->real_backend)point_rfft(fft,c->F,c->input);else point_fft(fft,c->F,c->input,0);
    #pragma omp parallel num_threads(fft->threads)
    {
        R a={0},b={0};
        #pragma omp for schedule(static)
        for(I j=0;j<L;j++){
            rmul(&a,&c->F[j].re,&c->F[j].re,bits,RN);rmul(&b,&c->F[j].im,&c->F[j].im,bits,RN);
            radd(&c->A[j].re,&a,&b,bits,RN);
        }
        rclear(&a);rclear(&b);
    }
    return c;
}
API void *point_cbc_prepare(FFT *fft,I n,I d,const I *perm,const double *gamma,const char *kernel){
    CBC *c=point_cbc_allocate(fft,n,d,perm,gamma);if(!c)return NULL;
    I m=c->m,bits=fft->bits;
    FILE *file=fopen(kernel,"r");if(!file){point_cbc_clear(c);return NULL;}
    char line[1024];Ball tmp={0};
    for(I j=-1;j<m;j++){
        if(!fgets(line,sizeof(line),file) || bparse(&tmp,line,bits+64)){bclear(&tmp);fclose(file);point_cbc_clear(c);return NULL;}
        rround(j<0?&c->psi0:c->psi+j,&tmp.mid,bits,RN);
    }
    bclear(&tmp);fclose(file);return point_cbc_finish(c);
}
API void *point_cbc_prepare_values(FFT *fft,I n,I d,const I *perm,const double *gamma,const R *values){
    CBC *c=point_cbc_allocate(fft,n,d,perm,gamma);if(!c)return NULL;
    rround(&c->psi0,values,fft->bits,RN);
    for(I j=0;j<c->m;j++)rround(c->psi+j,values+j+1,fft->bits,RN);
    return point_cbc_finish(c);
}
API void point_set_beta(CBC *c,const double *beta){
    for(I j=0;j<c->d;j++)rsetd(c->beta+j,beta[j]);
}
API void point_cbc_clear(CBC *c){if(!c)return;
    rfree(c->psi,c->m);rfree(c->gamma,c->d);rfree(c->beta,c->d);rfree(c->q,c->m);rfree(c->scores,c->m);
    rclear(&c->psi0);rclear(&c->sumpsi);
    point_vec_clear(c->F,c->fft->n);point_vec_clear(c->A,c->fft->n);
    point_vec_clear(c->input,c->fft->n);point_vec_clear(c->work,c->fft->n);
    point_vec_clear(c->product,c->fft->n);point_vec_clear(c->raw,c->fft->n);
    free(c->perm);free(c->lookup);free(c);
}
static I inverse_mod(I a,I n){I t=0,u=1,r=n,s=a;while(s){I q=r/s,v=t-q*u;t=u;u=v;v=r-q*s;r=s;s=v;}return t<0?t+n:t;}
static void mass(R *out,const R *q,I m,const R *q0,I bits){rsum(out,q,m,bits,RN);rscale(out,out,1);radd(out,out,q0,bits,RN);}
API int point_cbc(CBC *c,I *generators,Z *P,Z *updated,Z *gaps,double *stage_seconds,int audit){
    FFT *f=c->fft;I m=c->m,L=f->n,bits=f->bits;
    R q0={0},total={0},N={0},one={0},a={0},b={0},selected={0},minimum={0},gap={0},cumbeta={0};
    rseti(&N,c->n);rseti(&one,1);rset(&q0,&one);rset(&total,&N);
    rset(&cumbeta,&one);
    for(I s=0;s<c->d;s++){
        double clock=seconds();I w=0,equivalent=-1;
        if(s==0)rset(&selected,&c->sumpsi);
        else {
            if(s==1){if(f->real_backend)point_irfft(f,c->raw,c->A);else point_fft(f,c->raw,c->A,1);
                #pragma omp parallel for num_threads(f->threads) schedule(static)
                for(I j=0;j<m;j++){
                    if(j)radd(c->scores+j,&c->raw[j].re,&c->raw[L-m+j].re,bits,RN);
                    else rset(c->scores+j,&c->raw[j].re);
                }
            }else{
                #pragma omp parallel for num_threads(f->threads) schedule(static)
                for(I j=0;j<L;j++){
                    if(j<m)rset(&c->input[j].re,c->q+j);else zero_r(&c->input[j].re);
                    zero_r(&c->input[j].im);
                }
                if(f->real_backend)point_rfft(f,c->work,c->input);else point_fft(f,c->work,c->input,0);
                point_product(f,c->product,c->F,c->work);
                if(f->real_backend)point_irfft(f,c->raw,c->product);else point_fft(f,c->raw,c->product,1);
                #pragma omp parallel for num_threads(f->threads) schedule(static)
                for(I j=0;j<m;j++){
                    if(j<m-1)radd(c->scores+j,&c->raw[j].re,&c->raw[m+j].re,bits,RN);
                    else rset(c->scores+j,&c->raw[j].re);
                }
            }
            for(I j=1;j<m;j++)if(rcmp(c->scores+j,c->scores+w)<0)w=j;
            if(s==1){I inv=inverse_mod(c->perm[w],c->n);if(inv>c->n-inv)inv=c->n-inv;
                equivalent=c->lookup[inv];if(inv<c->perm[w]){I old=w;w=equivalent;equivalent=old;}}
            rset(&selected,c->scores+w);
            if(audit){int first=1;for(I j=0;j<m;j++)if(j!=w && j!=equivalent){
                if(first || rcmp(c->scores+j,&minimum)<0){rset(&minimum,c->scores+j);first=0;}}
                if(!first){rsub(&gap,&minimum,&selected,bits,RN);rset(&gaps[s].re,&gap);}}
            if(s==1){rmul(&selected,&selected,c->gamma,bits,RN);rmul(&b,c->beta,&c->sumpsi,bits,RN);radd(&selected,&selected,&b,bits,RN);}
        }
        generators[s]=c->perm[w];
        rmul(&a,&c->psi0,&q0,bits,RN);rscale(&b,&selected,1);radd(&a,&a,&b,bits,RN);
        rmul(&a,&a,c->gamma+s,bits,RN);rmul(&b,c->beta+s,&total,bits,RN);radd(&a,&a,&b,bits,RN);
        rmul(&cumbeta,&cumbeta,c->beta+s,bits,RN);
        rdiv(&a,&a,&N,bits,RN);rsub(&P[s].re,&a,&cumbeta,bits,RN);
        if(s<c->d-1 || audit){
            #pragma omp parallel num_threads(f->threads)
            {
                R factor={0};
                #pragma omp for schedule(static)
                for(I j=0;j<m;j++){
                    I k=w-j;if(k<0)k+=m;
                    rmul(&factor,c->gamma+s,c->psi+k,bits,RN);radd(&factor,&factor,c->beta+s,bits,RN);
                    if(s==0)rset(c->q+j,&factor);else rmul(c->q+j,c->q+j,&factor,bits,RN);
                }
                rclear(&factor);
            }
            rmul(&a,c->gamma+s,&c->psi0,bits,RN);radd(&a,&a,c->beta+s,bits,RN);rmul(&q0,&q0,&a,bits,RN);
            mass(&total,c->q,m,&q0,bits);rdiv(&a,&total,&N,bits,RN);rsub(&updated[s].re,&a,&cumbeta,bits,RN);
        }
        stage_seconds[s]=seconds()-clock;
    }
    rclear(&q0);rclear(&total);rclear(&N);rclear(&one);rclear(&a);rclear(&b);
    rclear(&selected);rclear(&minimum);rclear(&gap);rclear(&cumbeta);return 1;
}
API void point_fixed(CBC *c,const I *g,Z *out){
    I bits=c->fft->bits;R q0={0},one={0},a={0},N={0},total={0},cumbeta={0};
    rseti(&one,1);rset(&q0,&one);rseti(&N,c->n);
    rset(&cumbeta,&one);
    #pragma omp parallel num_threads(c->fft->threads)
    {
        R v={0},factor={0};
        #pragma omp for schedule(static)
        for(I j=0;j<c->m;j++){
            rset(&v,&one);
            for(I s=0;s<c->d;s++){
                I k=c->lookup[(c->perm[j]*g[s])%c->n];
                rmul(&factor,c->gamma+s,c->psi+k,bits,RN);radd(&factor,&factor,c->beta+s,bits,RN);
                rmul(&v,&v,&factor,bits,RN);
            }
            rset(c->q+j,&v);
        }
        rclear(&v);rclear(&factor);
    }
    for(I s=0;s<c->d;s++){rmul(&a,c->gamma+s,&c->psi0,bits,RN);radd(&a,&a,c->beta+s,bits,RN);rmul(&q0,&q0,&a,bits,RN);rmul(&cumbeta,&cumbeta,c->beta+s,bits,RN);}
    mass(&total,c->q,c->m,&q0,bits);rdiv(&total,&total,&N,bits,RN);rsub(&out->re,&total,&cumbeta,bits,RN);
    rclear(&q0);rclear(&one);rclear(&a);rclear(&N);rclear(&total);rclear(&cumbeta);
}
