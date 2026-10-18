// Copied from welvet-to-rust_n_cpp/welvet_cpp (code only; not documentation).
// Generated from native_arithmetic/tuned/outer.cpp.txt
#pragma once
// Product with contiguous output lanes: a*b or transpose(a)*b.
template<int DT> static void outer_impl(B&out,const B&a,const B&b,int m,int k,int n,bool at){
 std::fill(out.begin(),out.end(),0);constexpr int W=DT<2?(DT==0?4:8):DT<10?(1<<((DT-2)/2)):0;
 int ar=rowbytes(DT,at?m:k),br=rowbytes(DT,n);
 if constexpr(DT>=10){
  int words=(n+63)/64,levels=0;for(int t=k;t;t>>=1)levels++;
  for(int i=0;i<m;i++)for(int word=0;word<words;word++){
   uint64_t pos[64]={},neg[64]={};
   for(int q=0;q<k;q++){
    uint64_t g=at?code(a.data()+q*ar,i,m,DT):code(a.data()+i*ar,q,k,DT);if(!g)continue;
    uint64_t nz=readu(b.data()+q*br+word*8,8),sign=nz;if constexpr(DT==11)sign=readu(b.data()+q*br+(words+word)*8,8);else nz=~0ULL;if(g!=1)sign=~sign;
    uint64_t p=nz&sign;for(int level=0;level<levels&&p;level++){auto carry=pos[level]&p;pos[level]^=p;p=carry;}
    if constexpr(DT==11){p=nz&~sign;for(int level=0;level<levels&&p;level++){auto carry=neg[level]&p;neg[level]^=p;p=carry;}}
   }
   uint64_t gt=0,eq=~0ULL;for(int level=levels-1;level>=0;level--){uint64_t rhs=neg[level];if constexpr(DT==10)rhs=(((k+1)/2>>level)&1)?~0ULL:0;gt|=eq&pos[level]&~rhs;eq&=~(pos[level]^rhs);}
   uint64_t nz=DT==10?(gt|eq):~eq,sign=gt;if(word==words-1&&n%64){uint64_t mask=(1ULL<<(n%64))-1;nz&=mask;sign&=mask;}
   memcpy(out.data()+i*br+word*8,&nz,8);if constexpr(DT==11)memcpy(out.data()+i*br+(words+word)*8,&sign,8);
  }
 }else{
  for(int i=0;i<m;i++)for(int q=0;q<k;q++){
   const auto ap=a.data()+(at?q*ar+i*W:i*ar+q*W);auto bp=b.data()+q*br;auto dst=out.data()+i*br;int j=0;
   if constexpr(DT<2){
    using T=std::conditional_t<DT==0,float,double>;T g;memcpy(&g,ap,W);
    if constexpr(DT==0){auto gg=_mm256_set1_ps(g);for(;j+8<=n;j+=8)_mm256_storeu_ps((float*)dst+j,_mm256_fmadd_ps(gg,_mm256_loadu_ps((const float*)bp+j),_mm256_loadu_ps((const float*)dst+j)));}
    else{auto gg=_mm256_set1_pd(g);for(;j+4<=n;j+=4)_mm256_storeu_pd((double*)dst+j,_mm256_fmadd_pd(gg,_mm256_loadu_pd((const double*)bp+j),_mm256_loadu_pd((const double*)dst+j)));}
    for(;j<n;j++){T x,y;memcpy(&x,bp+j*W,W);memcpy(&y,dst+j*W,W);y=std::fma(g,x,y);memcpy(dst+j*W,&y,W);}
   }else{
    uint64_t g=readu(ap,W);__m256i gg;if constexpr(W==1)gg=_mm256_set1_epi8(g);else if constexpr(W==2)gg=_mm256_set1_epi16(g);else if constexpr(W==4)gg=_mm256_set1_epi32(g);else gg=_mm256_set1_epi64x(g);
    for(;j+32/W<=n;j+=32/W){auto v=_mm256_loadu_si256((const __m256i*)(bp+j*W)),y=_mm256_loadu_si256((const __m256i*)(dst+j*W));__m256i p;
     if constexpr(W==1){p=_mm256_or_si256(_mm256_and_si256(_mm256_mullo_epi16(v,gg),_mm256_set1_epi16(255)),_mm256_slli_epi16(_mm256_mullo_epi16(_mm256_srli_epi16(v,8),_mm256_srli_epi16(gg,8)),8));y=_mm256_add_epi8(y,p);}
     else if constexpr(W==2)y=_mm256_add_epi16(y,_mm256_mullo_epi16(v,gg));
     else if constexpr(W==4)y=_mm256_add_epi32(y,_mm256_mullo_epi32(v,gg));
     else{p=_mm256_add_epi64(_mm256_mul_epu32(v,gg),_mm256_slli_epi64(_mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(v,32),gg),_mm256_mul_epu32(v,_mm256_srli_epi64(gg,32))),32));y=_mm256_add_epi64(y,p);}
     _mm256_storeu_si256((__m256i*)(dst+j*W),y);
    }
    for(;j<n;j++){uint64_t v=readu(dst+j*W,W)+g*readu(bp+j*W,W);memcpy(dst+j*W,&v,W);}
   }
  }
 }
}
static void outer(B&o,const B&a,const B&b,int m,int k,int n,int dt,bool at){switch(dt){
#define CASE(D) case D:outer_impl<D>(o,a,b,m,k,n,at);break;
CASE(0) CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11)
#undef CASE
}}
