#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>
#include <math.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include "nlohmann/json.hpp"

#define RAND_BUF_LEN            2
#define ONE_OVER_C0             3.335640951981520e-12f
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

typedef uint64_t  RandType;

using json = nlohmann::json;

struct float4 {
    float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
    float4() {}
    float4(float v) : x(v), y(v), z(v), w(v) {}
    float4(float x0, float y0, float z0, float w0) : x(x0), y(y0), z(z0), w(w0) {}
    void set(float x0, float y0, float z0, float w0)  {
        x = x0, y = y0, z = z0, w = w0;
    }
    float4& operator=(float4 a) {
        set(a.x, a.y, a.z, a.w);
        return *this;
    }
};

struct dim4 {
    uint32_t x = 0u, y = 0u, z = 0u, w = 0u;
    dim4() {}
    dim4(uint32_t v) : x(v), y(v), z(v), w(v) {}
    dim4(uint32_t x0, uint32_t y0, uint32_t z0, uint32_t w0) : x(x0), y(y0), z(z0), w(w0) {}
};

struct short4 {
    int16_t x = 0, y = 0, z = 0, w = 0;
    short4() {}
    short4(int16_t v) : x(v), y(v), z(v), w(v) {}
    short4(int16_t x0, int16_t y0, int16_t z0, int16_t w0) : x(x0), y(y0), z(z0), w(w0) {}
};

struct mcx_medium {
    float mua = 0.f, mus = 0.f, g = 1.f, n = 1.f;
    mcx_medium() {}
    mcx_medium(float mua0, float mus0) : mua(mua0), mus(mus0) {}
    mcx_medium(float mua0, float mus0, float g0, float n0) : mua(mua0), mus(mus0), g(g0), n(n0) {}
};

struct mcx_rand { // per thread
    RandType t[RAND_BUF_LEN];

    mcx_rand(dim4 seed) {
        t[0] = (uint64_t)seed.x << 32 | seed.y;
        t[1] = (uint64_t)seed.z << 32 | seed.w;
    }
    float rand01() { //< advance random state, return a uniformly 0-1 distributed float random number
        union {
            uint64_t i;
            float f[2];
            uint  u[2];
        } s1;
        const uint64_t s0 = t[1];
        s1.i = t[0];
        t[0] = s0;
        s1.i ^= s1.i << 23; // a
        t[1] = s1.i ^ s0 ^ (s1.i >> 18) ^ (s0 >> 5); // b, c
        s1.i = t[1] + s0;
        s1.u[0] = 0x3F800000U | (s1.u[0] >> 9);

        return s1.f[0] - 1.0f;
    }
    float next_scat_len() {
        return -logf(rand01() + FLT_EPSILON);
    }
};

template<class T>
class mcx_volume { // shared, read-only
    dim4 size;
    uint64_t dimxy = 0, dimxyz = 0, dimxyzt = 0;
    T* vol = nullptr;

  public:
    mcx_volume(uint32_t Nx, uint32_t Ny, uint32_t Nz, uint32_t Nt = 1) {
        size = dim4(Nx, Ny, Nz, Nt);
        dimxy = Nx * Ny;
        dimxyz = dimxy * Ny;
        dimxyzt = dimxyz * Nt;
        vol = new T[dimxyzt]();

        for (uint64_t i = 0; i < dimxyzt; i++) {
            vol[i] = 1;
        }
    }
    ~mcx_volume () {
        size = dim4(0);
        delete [] vol;
        vol = nullptr;
    }
    void loadfromjnii (std::string fname) {
        std::ifstream inputjnii(fname);
        json jnii;
        inputjnii >> jnii;
        size.x = jnii["NIFTIData"]["_ArraySize_"][0];
        size.y = jnii["NIFTIData"]["_ArraySize_"][1];
        size.z = jnii["NIFTIData"]["_ArraySize_"][2];
    }
    int64_t index(short ix, short iy, short iz, int it = 0) { // when outside the volume, return -1, otherwise, return 1d index
        return !(ix < 0 || iy < 0 || iz < 0 || ix >= (short)size.x || iy >= (short)size.y || iz >= (short)size.z || it >= (int)size.w) ? (it * dimxyz + iz * dimxy + iy * size.x + ix) : -1;
    }
    T& get(int64_t idx) const  { // must be inside the volume
        return vol[idx];
    }
    void add(T val, int64_t idx) const  {
        vol[idx] += val;
    }
};

struct mcx_photon { // per thread
    float4 pos /*{x,y,z,w}*/, vec /*{vx,vy,vz,nscat}*/, rvec /*1/vx,1/vy,1/vz,*/, len /*{pscat,t,pathlen,p0}*/;
    short4 ipos /*{ix,iy,iz,flipdir}*/;
    int64_t lastvoxelidx;
    int mediaid;

    mcx_photon(std::vector<float> p0, std::vector<float> v0) { // constructor
        pos.set(p0[0], p0[1], p0[2], p0.size() > 3 ? p0[3] : 1.f);
        vec.set(v0[0], v0[1], v0[2], 0.f);
        rvec.set(1.f / v0[0], 1.f / v0[1], 1.f / v0[2], 1.f);
        len.set(NAN, 0.f, 0.f, 0.f);
        ipos = short4((short)p0[0], (short)p0[1], (short)p0[2], -1);
        lastvoxelidx = -1;
        mediaid = 0;
    }
    void run(mcx_volume<int>& invol, mcx_volume<float>& outvol, mcx_medium props[], mcx_rand& ran) { // main function to run a single photon from lunch to termination
        lastvoxelidx = outvol.index(ipos.x, ipos.y, ipos.z, 0);
        mediaid = invol.get(lastvoxelidx);
        len.x = ran.next_scat_len();

        while (1) {
            if (sprint(invol, outvol, props)) {
                break;
            }

            scatter(props[mediaid], ran);
        }
    }
    int sprint(mcx_volume<int>& invol, mcx_volume<float>& outvol, mcx_medium props[]) { // run from one scattering site to the next, return 1 when terminate
        while (len.x > 0.f) {
            int64_t newvoxelid = step(invol, props[mediaid]);

            if (newvoxelid > 0 && newvoxelid != lastvoxelidx) { // only save when moving out of a voxel
                save(outvol);
                lastvoxelidx = newvoxelid; // save last saving site
                mediaid = invol.get(lastvoxelidx);
            } else if (newvoxelid < 0) {
                return 1;
            }
        }

        return 0;
    }
    int64_t step(mcx_volume<int>& invol, mcx_medium prop) {
        float dist, htime[3];

        htime[0] = fabsf((ipos.x + (vec.x > 0.f) - pos.x) * rvec.x);  //< time-of-flight to hit the wall in each direction
        htime[1] = fabsf((ipos.y + (vec.y > 0.f) - pos.y) * rvec.y);
        htime[2] = fabsf((ipos.z + (vec.z > 0.f) - pos.z) * rvec.z);
        dist = fminf(fminf(htime[0], htime[1]), htime[2]);            //< get the direction with the smallest time-of-flight
        ipos.w = (dist == htime[0] ? 0 : (dist == htime[1] ? 1 : 2)); //< determine which axis plane the photon crosses

        htime[0] = dist * prop.mus;
        htime[0] = fminf(htime[0], len.x);
        htime[1] = (htime[0] != len.x); // is continue next voxel?
        dist = (prop.mus == 0.f) ? dist : (htime[0] / prop.mus);
        pos.set(pos.x + dist * vec.x, pos.y + dist * vec.y, pos.z + dist * vec.z, expf(-prop.mua * dist));

        len.x -= htime[0];
        len.y += dist * prop.n * ONE_OVER_C0;
        len.z += dist;

        if (htime[1] > 0.f) { // photon need to move to next voxel
            (ipos.w == 0) ? (ipos.x += (vec.x > 0.f ? 1 : -1)) :
            ((ipos.w == 1) ? ipos.y += (vec.y > 0.f ? 1 : -1) :
                                       (ipos.z += (vec.z > 0.f ? 1 : -1)));
            return invol.index(ipos.x, ipos.y, ipos.z);
        }

        return lastvoxelidx;
    }
    void save(mcx_volume<float>& outvol) {
        outvol.add(len.w - pos.w, lastvoxelidx);
        len.w = pos.w;
    }
    void scatter(mcx_medium& prop, mcx_rand& ran) {
        float tmp0, sphi, cphi, theta, stheta, ctheta;
        len.x = ran.next_scat_len();

        tmp0 = (2.f * M_PI) * ran.rand01(); //next arimuth angle
        sincosf(tmp0, &sphi, &cphi);

        if (fabsf(prop.g) > FLT_EPSILON) { //< if prop.g is too small, the distribution of theta is bad
            tmp0 = (1.f - prop.g * prop.g) / (1.f - prop.g + 2.f * prop.g * ran.rand01());
            tmp0 *= tmp0;
            tmp0 = (1.f + prop.g * prop.g - tmp0) / (2.f * prop.g);
            tmp0 = fmaxf(-1.f, fminf(1.f, tmp0));

            theta = acosf(tmp0);
            stheta = sinf(theta);
            ctheta = tmp0;
        } else {
            theta = acosf(2.f * ran.rand01() - 1.f);
            sincosf(theta, &stheta, &ctheta);
        }

        rotatevector(stheta, ctheta, sphi, cphi);
        rvec.set(1.f / vec.x, 1.f / vec.y, 1.f / vec.z, 1.f);
    }
    float reflectcoeff(float n1, float n2, int flipdir) {
        float Icos = fabsf((flipdir == 0) ? vec.x : (flipdir == 1 ? vec.y : vec.z));
        float tmp0 = n1 * n1;
        float tmp1 = n2 * n2;
        float tmp2 = 1.f - tmp0 / tmp1 * (1.f - Icos * Icos); /** 1-[n1/n2*sin(si)]^2 = cos(ti)^2*/

        if (tmp2 > 0.f) { //< partial reflection
            float Re, Im, Rtotal;
            Re = tmp0 * Icos * Icos + tmp1 * tmp2;
            tmp2 = sqrtf(tmp2); /** to save one sqrt*/
            Im = 2.f * n1 * n2 * Icos * tmp2;
            Rtotal = (Re - Im) / (Re + Im); /** Rp*/
            Re = tmp1 * Icos * Icos + tmp0 * tmp2 * tmp2;
            Rtotal = (Rtotal + (Re - Im) / (Re + Im)) * 0.5f; /** (Rp+Rs)/2*/
            return Rtotal;
        }

        return 1.f;  //< total reflection
    }
    void transmit(float n1, float n2, int flipdir) {
        float tmp0 = n1 / n2;

        vec.x *= tmp0;
        vec.y *= tmp0;
        vec.z *= tmp0;
        (flipdir == 0) ?
        (vec.x = ((tmp0 = vec.y * vec.y + vec.z * vec.z) < 1.f) ? sqrtf(1.f - tmp0) * ((vec.x > 0.f) - (vec.x < 0.f)) : 0.f) :
        ((flipdir == 1) ?
         (vec.y = ((tmp0 = vec.x * vec.x + vec.z * vec.z) < 1.f) ? sqrtf(1.f - tmp0) * ((vec.y > 0.f) - (vec.y < 0.f)) : 0.f) :
         (vec.z = ((tmp0 = vec.x * vec.x + vec.y * vec.y) < 1.f) ? sqrtf(1.f - tmp0) * ((vec.z > 0.f) - (vec.z < 0.f)) : 0.f));
        tmp0 = 1.f / sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        vec.x *= tmp0;
        vec.y *= tmp0;
        vec.z *= tmp0;
    }
    float reflect(float n1, float n2, int flipdir, mcx_rand& ran) {
        float Rtotal = reflectcoeff(n1, n2, flipdir);

        if (ran.rand01() > Rtotal) {
            (flipdir == 0) ? (vec.x = -vec.x) : ((flipdir == 1) ? (vec.y = -vec.y) : (vec.z = -vec.z)) ;
        } else {
            transmit(n1, n2, flipdir);
        }

        return Rtotal;
    }
    void rotatevector(float stheta, float ctheta, float sphi, float cphi) {
        if ( vec.z > -1.f + FLT_EPSILON && vec.z < 1.f - FLT_EPSILON ) {
            float tmp0 = 1.f - vec.z * vec.z;
            float tmp1 = stheta / sqrtf(tmp0);
            vec.set(
                tmp1 * (vec.x * vec.z * cphi - vec.y * sphi) + vec.x * ctheta,
                tmp1 * (vec.y * vec.z * cphi + vec.x * sphi) + vec.y * ctheta,
                -tmp1 * tmp0 * cphi                          + vec.z * ctheta,
                vec.w
            );
        } else {
            vec.set(stheta * cphi, stheta * sphi, (vec.z > 0.f) ? ctheta : -ctheta, vec.w);
        }

        float tmp0 = 1.f / sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        vec.x *= tmp0;
        vec.y *= tmp0;
        vec.z *= tmp0;
    }
};

int main(int argn, char* argv[]) {
    if (argn == 1) {
        std::cout << "format: umcx input.json" << std::endl;
        return 0;
    }

    std::ifstream inputjson(argv[1]);
    json cfg;
    inputjson >> cfg;

    mcx_volume<int> inputvol(cfg["Domain"]["Dim"][0], cfg["Domain"]["Dim"][1], cfg["Domain"]["Dim"][2]);
    mcx_volume<float> outputvol(cfg["Domain"]["Dim"][0], cfg["Domain"]["Dim"][1], cfg["Domain"]["Dim"][2]);
    mcx_medium* prop = new mcx_medium[cfg["Domain"]["Media"].size()];

    for (uint32_t i = 0; i < cfg["Domain"]["Media"].size(); i++) {
        prop[i] = mcx_medium(cfg["Domain"]["Media"][i]["mua"], cfg["Domain"]["Media"][i]["mus"], cfg["Domain"]["Media"][i]["g"], cfg["Domain"]["Media"][i]["n"]);
    }

    for (uint64_t i = 0; i < cfg["Session"]["Photons"]; i++) {
        mcx_rand ran(dim4(std::rand(), std::rand(), std::rand(), std::rand()));
        mcx_photon p(cfg["Optode"]["Source"]["Pos"].get<std::vector<float>>(), cfg["Optode"]["Source"]["Dir"].get<std::vector<float>>());
        p.run(inputvol, outputvol, prop, ran);
    }

    delete [] prop;
    return 0;
}
