// Copied from welvet-to-rust_n_cpp/welvet_cpp (code only; not documentation).
#pragma once
#include <immintrin.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <type_traits>
using Bytes=std::vector<uint8_t>;
inline uint64_t readu(const uint8_t*p,int w){uint64_t q=0;memcpy(&q,p,w);return q;}
inline int width(int dt){static int v[]={4,8,1,1,2,2,4,4,8,8,0,0};return v[dt];}
inline int rowbytes(int dt,int n){return dt<10?n*width(dt):(n+63)/64*8*(dt==11?2:1);}
inline uint64_t code(const uint8_t*p,int i,int n,int dt){
 if(dt<10)return readu(p+i*width(dt),width(dt));
 auto bit=[&](int plane){return (readu(p+(plane*((n+63)/64)+i/64)*8,8)>>(i%64))&1;};
 if(dt==10)return bit(0)?1:uint64_t(-1);
 return bit(0)?(bit(1)?1:uint64_t(-1)):0;
}
inline int64_t value(const uint8_t*p,int i,int n,int dt){uint64_t q=code(p,i,n,dt);int bits=width(dt)*8;if(dt<10&&dt%2==0&&bits<64&&q&(1ULL<<(bits-1)))q|=~((1ULL<<bits)-1);int64_t v;memcpy(&v,&q,8);return v;}
inline void putcode(uint8_t*p,int i,int n,int dt,uint64_t q){
 if(dt<10){memcpy(p+i*width(dt),&q,width(dt));return;}
 auto set=[&](int plane,bool on){auto at=p+(plane*((n+63)/64)+i/64)*8;uint64_t bit=1ULL<<(i%64),v=readu(at,8);v=on?(v|bit):(v&~bit);memcpy(at,&v,8);};
 if(dt==10)set(0,q==1);else{set(0,q!=0);set(1,q==1);}
}
inline __m256i mul64(__m256i a,__m256i b){return _mm256_add_epi64(_mm256_mul_epu32(a,b),_mm256_slli_epi64(_mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a,32),b),_mm256_mul_epu32(a,_mm256_srli_epi64(b,32))),32));}
template<int DT> inline __m256i load4(const uint8_t*p,int i,int n){
 if constexpr(DT==2)return _mm256_cvtepi8_epi64(_mm_cvtsi32_si128(uint32_t(readu(p+i,4))));
 if constexpr(DT==3)return _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(uint32_t(readu(p+i,4))));
 if constexpr(DT==4)return _mm256_cvtepi16_epi64(_mm_loadl_epi64((const __m128i*)(p+i*2)));
 if constexpr(DT==5)return _mm256_cvtepu16_epi64(_mm_loadl_epi64((const __m128i*)(p+i*2)));
 if constexpr(DT==6)return _mm256_cvtepi32_epi64(_mm_loadu_si128((const __m128i*)(p+i*4)));
 if constexpr(DT==7)return _mm256_cvtepu32_epi64(_mm_loadu_si128((const __m128i*)(p+i*4)));
 if constexpr(DT==8||DT==9)return _mm256_loadu_si256((const __m256i*)(p+i*8));
 if constexpr(DT>=10){auto lanes=_mm256_set_epi64x(8,4,2,1),zero=_mm256_setzero_si256(),ones=_mm256_set1_epi64x(1);auto bits=_mm256_set1_epi64x(readu(p+(i/64)*8,8)>>(i%64));auto nz=_mm256_cmpeq_epi64(_mm256_and_si256(bits,lanes),zero);
 if constexpr(DT==10)return _mm256_or_si256(nz,ones);
 auto sign=_mm256_set1_epi64x(readu(p+(((n+63)/64)+(i/64))*8,8)>>(i%64));sign=_mm256_cmpeq_epi64(_mm256_and_si256(sign,lanes),zero);return _mm256_andnot_si256(nz,_mm256_or_si256(sign,ones));}
}
template<class T,int DT> void axpy(T*dst,const uint8_t*src,T g,int n,bool reference){
 int i=0;if(!reference){
 if constexpr(DT<2){constexpr int N=DT==0?8:4;
 for(;i+N<=n;i+=N){if constexpr(DT==0)_mm256_storeu_ps(dst+i,_mm256_fmadd_ps(_mm256_set1_ps(g),_mm256_loadu_ps((const float*)src+i),_mm256_loadu_ps(dst+i)));else _mm256_storeu_pd(dst+i,_mm256_fmadd_pd(_mm256_set1_pd(g),_mm256_loadu_pd((const double*)src+i),_mm256_loadu_pd(dst+i)));}}
 else {auto gg=_mm256_set1_epi64x(g);for(;i+4<=n;i+=4){auto v=load4<DT>(src,i,n);__m256i product;if constexpr(DT>=10){auto zero=_mm256_setzero_si256(),negative=_mm256_cmpgt_epi64(zero,v);product=_mm256_sub_epi64(_mm256_xor_si256(gg,negative),negative);if constexpr(DT==11)product=_mm256_andnot_si256(_mm256_cmpeq_epi64(v,zero),product);}else product=mul64(v,gg);_mm256_storeu_si256((__m256i*)(dst+i),_mm256_add_epi64(_mm256_loadu_si256((const __m256i*)(dst+i)),product));}}}
 for(;i<n;i++){if constexpr(DT<2){T x;memcpy(&x,src+i*sizeof(T),sizeof(T));dst[i]=std::fma(g,x,dst[i]);}else {uint64_t q=uint64_t(dst[i])+uint64_t(g)*uint64_t(value(src,i,n,DT));memcpy(dst+i,&q,8);}}
}
inline int64_t clip(int64_t g){return std::max(-(1LL<<30),std::min(1LL<<30,(long long)g));}
__attribute__((always_inline)) inline void update_int(uint8_t*p,int i,int n,int dt,int64_t g,int64_t&r){
 int64_t u=r-clip(g);int64_t quantum=dt==10?4194304:2097152;
 int64_t delta=(std::abs(u)+quantum/2-1)/quantum;if(u<0)delta=-delta;
 uint64_t q=code(p,i,n,dt),max=~0ULL,bias=0;
 if(dt>=10){q=dt==10?(q==1):uint64_t(int64_t(q)+1);max=dt==10?1:2;}
 else{int bits=width(dt)*8;max=bits==64?~0ULL:(1ULL<<bits)-1;if(dt%2==0)bias=1ULL<<(bits-1);q^=bias;}
 uint64_t before=q;
 if(delta>0)q=max-q<uint64_t(delta)?max:q+delta;else if(delta<0)q=q<uint64_t(-delta)?0:q-uint64_t(-delta);
 int64_t applied=q>=before?int64_t(q-before):-int64_t(before-q);r=u-applied*quantum;
 if((q==0&&r<0)||(q==max&&r>0))r=0;
 uint64_t out=dt==10?(q?1:uint64_t(-1)):dt==11?uint64_t(int64_t(q)-1):(q^bias);putcode(p,i,n,dt,out);
}
template<class T> void update_float(uint8_t*p,T g,double&r){T old;memcpy(&old,p,sizeof(T));double change=r-double(g)/32;T next=T(double(old)+change);double z=double(next)-double(old);r=(double(old)-(double(next)-z))+(change-z);memcpy(p,&next,sizeof(T));}
