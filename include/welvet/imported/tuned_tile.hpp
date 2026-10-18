// Copied from welvet-to-rust_n_cpp/welvet_cpp (code only; not documentation).
#pragma once
template<int DT> static void tiled_impl(B&out,const B&a,const B&b,int m,int k,int n){
 if constexpr(DT>=10){product(out,a,b,m,k,n,DT);return;}else{
 constexpr int W=DT<2?(DT==0?4:8):(1<<((DT-2)/2)),L=32/W;int ar=k*W;
 for(int row=0;row<m;row++)for(int o=0;o<n;o+=4){
 if(o+4>n){for(int j=o;j<n;j++){uint64_t v=0;if constexpr(DT==0){float x=dot_ps((const float*)(a.data()+row*ar),(const float*)(b.data()+j*ar),k,false);memcpy(&v,&x,4);}else if constexpr(DT==1){double x=dot_pd((const double*)(a.data()+row*ar),(const double*)(b.data()+j*ar),k,false);memcpy(&v,&x,8);}else v=dot_int<W>(a.data()+row*ar,b.data()+j*ar,k,false);memcpy(out.data()+(row*n+j)*W,&v,W);}break;}
 if constexpr(DT==0){
 __m256 acc[4];for(auto&v:acc)v=_mm256_setzero_ps();int q=0;
 for(;q+L<=k;q+=L){auto x=_mm256_loadu_ps((const float*)(a.data()+row*ar)+q);for(int j=0;j<4;j++)acc[j]=_mm256_fmadd_ps(x,_mm256_loadu_ps((const float*)(b.data()+(o+j)*ar)+q),acc[j]);}
 for(int j=0;j<4;j++){float v[L];_mm256_storeu_ps(v,acc[j]);for(int step=L/2;step;step/=2)for(int z=0;z<step;z++)v[z]+=v[z+step];float s=v[0];for(int z=q;z<k;z++)s=std::fma(((const float*)(a.data()+row*ar))[z],((const float*)(b.data()+(o+j)*ar))[z],s);memcpy(out.data()+(row*n+o+j)*W,&s,W);}
 }
 else if constexpr(DT==1){
 __m256d acc[4];for(auto&v:acc)v=_mm256_setzero_pd();int q=0;
 for(;q+L<=k;q+=L){auto x=_mm256_loadu_pd((const double*)(a.data()+row*ar)+q);for(int j=0;j<4;j++)acc[j]=_mm256_fmadd_pd(x,_mm256_loadu_pd((const double*)(b.data()+(o+j)*ar)+q),acc[j]);}
 for(int j=0;j<4;j++){double v[L];_mm256_storeu_pd(v,acc[j]);for(int step=L/2;step;step/=2)for(int z=0;z<step;z++)v[z]+=v[z+step];double s=v[0];for(int z=q;z<k;z++)s=std::fma(((const double*)(a.data()+row*ar))[z],((const double*)(b.data()+(o+j)*ar))[z],s);memcpy(out.data()+(row*n+o+j)*W,&s,W);}
 }
else {__m256i acc[4];for(auto&v:acc)v=_mm256_setzero_si256();int q=0;
 for(;q+L<=k;q+=L){auto x=_mm256_loadu_si256((const __m256i*)(a.data()+row*ar+q*W));for(int j=0;j<4;j++){auto y=_mm256_loadu_si256((const __m256i*)(b.data()+(o+j)*ar+q*W));__m256i p;
 if constexpr(W==1){p=_mm256_or_si256(_mm256_and_si256(_mm256_mullo_epi16(x,y),_mm256_set1_epi16(255)),_mm256_slli_epi16(_mm256_mullo_epi16(_mm256_srli_epi16(x,8),_mm256_srli_epi16(y,8)),8));acc[j]=_mm256_add_epi8(acc[j],p);}
 else if constexpr(W==2)acc[j]=_mm256_add_epi16(acc[j],_mm256_mullo_epi16(x,y));else if constexpr(W==4)acc[j]=_mm256_add_epi32(acc[j],_mm256_mullo_epi32(x,y));else{p=_mm256_add_epi64(_mm256_mul_epu32(x,y),_mm256_slli_epi64(_mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(x,32),y),_mm256_mul_epu32(x,_mm256_srli_epi64(y,32))),32));acc[j]=_mm256_add_epi64(acc[j],p);}
 }}
 for(int j=0;j<4;j++){uint8_t v[32];_mm256_storeu_si256((__m256i*)v,acc[j]);uint64_t sum=0;for(int z=0;z<L;z++)sum+=rd(v+z*W,W);for(int z=q;z<k;z++)sum+=rd(a.data()+row*ar+z*W,W)*rd(b.data()+(o+j)*ar+z*W,W);memcpy(out.data()+(row*n+o+j)*W,&sum,W);}
 }
 }}}
static void tiled(B&o,const B&a,const B&b,int m,int k,int n,int dt){switch(dt){
case 0:tiled_impl<0>(o,a,b,m,k,n);break;
case 1:tiled_impl<1>(o,a,b,m,k,n);break;
case 2:tiled_impl<2>(o,a,b,m,k,n);break;
case 3:tiled_impl<3>(o,a,b,m,k,n);break;
case 4:tiled_impl<4>(o,a,b,m,k,n);break;
case 5:tiled_impl<5>(o,a,b,m,k,n);break;
case 6:tiled_impl<6>(o,a,b,m,k,n);break;
case 7:tiled_impl<7>(o,a,b,m,k,n);break;
case 8:tiled_impl<8>(o,a,b,m,k,n);break;
case 9:tiled_impl<9>(o,a,b,m,k,n);break;
case 10:tiled_impl<10>(o,a,b,m,k,n);break;
case 11:tiled_impl<11>(o,a,b,m,k,n);break;
}}
