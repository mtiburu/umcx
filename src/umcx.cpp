//////////////////////////////////////////////////////////////////////////////////////////////////
///  \mainpage uMCX: readable, portable, hackable and massively-parallel photon simulator
///  \copyright Qianqian Fang <q.fang at neu.edu>, 2024-2025
///  \section slicense License
///          GPL v3, see LICENSE.txt for details
//////////////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>
#include <math.h>
#include <chrono>

#ifdef _OPENMP
    #include <omp.h>
#endif
#include "nlohmann/json.hpp"

#define ONE_OVER_C0             3.335640951981520e-12f
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

using json = nlohmann::ordered_json;
#pragma omp declare target
/// basic data type: float4 class
/** float4 data type has 4x float elements {x,y,z,w}, used for representing photon states */
struct float4 {
    float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
};
/// basic data type: dim4 class
/** dim4 is used to represent array dimensions, with 4x uint32_t members {x,y,z,w} */
struct dim4 {
    uint32_t x = 0u, y = 0u, z = 0u, w = 0u;
};
/// basic data type: short4 class
/**  */
struct short4 {
    int16_t x = 0, y = 0, z = 0, w = 0;
};
/// Volumetric optical properties
/** MCX_medium has 4 float members, mua (absorption coeff., 1/mm), mus (scattering coeff., 1/mm), g (anisotropy) and n (ref. coeff.)*/
struct MCX_medium {
    float mua = 0.f, mus = 0.f, g = 1.f, n = 1.f;
};
#pragma omp end declare target
/// MCX_volume class manages input and output volume
/** */
template<class T>
class MCX_volume { // shared, read-only
    dim4 size;
    uint64_t dimxy = 0, dimxyz = 0, dimxyzt = 0;
    T* vol = nullptr;

  public:
    MCX_volume(uint32_t Nx, uint32_t Ny, uint32_t Nz, uint32_t Nt = 1, T value = 0.0) {
        size = (dim4) {
            Nx, Ny, Nz, Nt
        };
        dimxy = Nx * Ny;
        dimxyz = dimxy * Ny;
        dimxyzt = dimxyz * Nt;
        vol = new T[dimxyzt]();

        if (value != 0)
            for (uint64_t i = 0; i < dimxyzt; i++) {
                vol[i] = value;
            }
    }
    ~MCX_volume () {
        size = (dim4) {
            0, 0, 0, 0
        };
        delete [] vol;
        vol = nullptr;
    }
    void loadfromjnii (std::string fname) {
        std::ifstream inputjnii(fname);
        json jnii;
        inputjnii >> jnii;
        size = dim4(jnii["NIFTIData"]["_ArraySize_"][0], jnii["NIFTIData"]["_ArraySize_"][1], jnii["NIFTIData"]["_ArraySize_"][2]);
    }
    int64_t index(short ix, short iy, short iz, int it = 0) { // when outside the volume, return -1, otherwise, return 1d index
        return !(ix < 0 || iy < 0 || iz < 0 || ix >= (short)size.x || iy >= (short)size.y || iz >= (short)size.z || it >= (int)size.w) ? (it * dimxyz + iz * dimxy + iy * size.x + ix) : -1;
    }
    T& get(int64_t idx) const  { // must be inside the volume
        return vol[idx];
    }
    void add(T val, int64_t idx) const  {
        #pragma omp atomic
        vol[idx] += val;
    }
    T* buffer() const  {
        return vol;
    }
    uint64_t count() const  {
        return dimxyzt;
    }
};
#pragma omp declare target
/// MCX_rand provides the xorshift128p random number generator
/**  */
struct MCX_rand { // per thread
    uint64_t t[2];

    MCX_rand(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
        t[0] = (uint64_t)s0 << 32 | s1;
        t[1] = (uint64_t)s2 << 32 | s3;
    }
    float rand01() { //< advance random state, return a uniformly 0-1 distributed float random number
        union {
            uint64_t i;
            float f[2];
            uint32_t u[2];
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
/// MCX_photon class performs MC simulation of a single photon
/** */
struct MCX_photon { // per thread
    float4 pos /*{x,y,z,w}*/, vec /*{vx,vy,vz,nscat}*/, rvec /*1/vx,1/vy,1/vz,unused*/, len /*{pscat,t,pathlen,p0}*/;
    short4 ipos /*{ix,iy,iz,flipdir}*/;
    int64_t lastvoxelidx;
    int mediaid;

    MCX_photon(float4& p0, float4& v0) { // constructor
        pos = p0;
        vec = v0;
        rvec = (float4) {
            1.f / v0.x, 1.f / v0.y, 1.f / v0.z, 1.f
        };
        len = (float4) {
            NAN, 0.f, 0.f, pos.w
        };
        ipos = (short4) {
            (short)p0.x, (short)p0.y, (short)p0.z, -1
        };
        lastvoxelidx = -1;
        mediaid = 0;
    }
    void run(MCX_volume<int>& invol, MCX_volume<float>& outvol, MCX_medium props[], MCX_rand& ran) { // main function to run a single photon from lunch to termination
        lastvoxelidx = outvol.index(ipos.x, ipos.y, ipos.z, 0);
        mediaid = invol.get(lastvoxelidx);
        len.x = ran.next_scat_len();

        while (1) {
            if (sprint(invol, outvol, props, ran)) {
                break;
            }

            scatter(props[mediaid], ran);
        }
    }
    int sprint(MCX_volume<int>& invol, MCX_volume<float>& outvol, MCX_medium props[], MCX_rand& ran) { // run from one scattering site to the next, return 1 when terminate
        while (len.x > 0.f) {
            int64_t newvoxelid = step(invol, props[mediaid]);

            if (newvoxelid != lastvoxelidx) { // only save when moving out of a voxel
                save(outvol);
                int newmediaid = ((newvoxelid >= 0) ? invol.get(newvoxelid) : 0);

                if (props[mediaid].n != props[newmediaid].n) {
                    if (reflect(props[mediaid].n, props[newmediaid].n, ran) && (newvoxelid < 0 || newmediaid == 0)) {
                        return 1;
                    }
                } else if (newvoxelid < 0 || newmediaid == 0) {
                    return 1;
                }

                lastvoxelidx = newvoxelid; // save last saving site
                mediaid = newmediaid;
            } else {
                return 0;
            }
        }

        return 0;
    }
    int64_t step(MCX_volume<int>& invol, MCX_medium prop) {
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
        pos = (float4) {
            pos.x + dist* vec.x, pos.y + dist* vec.y, pos.z + dist* vec.z, pos.w* expf(-prop.mua * dist)
        };

        len.x -= htime[0];
        len.y += dist * prop.n * ONE_OVER_C0;
        len.z += dist;

        if (htime[1] > 0.f) { // photon need to move to next voxel
            (ipos.w == 0) ? (ipos.x += (vec.x > 0.f ? 1 : -1)) :
            ((ipos.w == 1) ? ipos.y += (vec.y > 0.f ? 1 : -1) :
                                       (ipos.z += (vec.z > 0.f ? 1 : -1))); // update ipos.xyz based on ipos.w = flipdir
            return invol.index(ipos.x, ipos.y, ipos.z);
        }

        return lastvoxelidx;
    }
    void save(MCX_volume<float>& outvol) {
        outvol.add(len.w - pos.w, lastvoxelidx);
        len.w = pos.w;
    }
    void scatter(MCX_medium& prop, MCX_rand& ran) {
        float tmp0, sphi, cphi, theta, stheta, ctheta;
        len.x = ran.next_scat_len();

        tmp0 = (2.f * M_PI) * ran.rand01(); //next arimuth angle
        //sphi = sinf(tmp0);
        //cphi = cosf(tmp0);
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
            //stheta = sinf(theta);
            //ctheta = cosf(theta);
            sincosf(theta, &stheta, &ctheta);
        }

        rotatevector(stheta, ctheta, sphi, cphi);
        rvec = (float4) {
            1.f / vec.x, 1.f / vec.y, 1.f / vec.z, 1.f
        };
        vec.w++; // stops at 16777216 due to finite precision
    }
    float reflectcoeff(float n1, float n2) {
        float Icos = fabsf((ipos.w == 0) ? vec.x : (ipos.w == 1 ? vec.y : vec.z));
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
    void transmit(float n1, float n2) {
        float tmp0 = n1 / n2;

        vec.x *= tmp0;
        vec.y *= tmp0;
        vec.z *= tmp0;
        (ipos.w == 0) ?
        (vec.x = ((tmp0 = vec.y * vec.y + vec.z * vec.z) < 1.f) ? sqrtf(1.f - tmp0) * ((vec.x > 0.f) - (vec.x < 0.f)) : 0.f) :
        ((ipos.w == 1) ?
         (vec.y = ((tmp0 = vec.x * vec.x + vec.z * vec.z) < 1.f) ? sqrtf(1.f - tmp0) * ((vec.y > 0.f) - (vec.y < 0.f)) : 0.f) :
         (vec.z = ((tmp0 = vec.x * vec.x + vec.y * vec.y) < 1.f) ? sqrtf(1.f - tmp0) * ((vec.z > 0.f) - (vec.z < 0.f)) : 0.f));
        tmp0 = 1.f / sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        vec.x *= tmp0;
        vec.y *= tmp0;
        vec.z *= tmp0;
    }
    int reflect(float n1, float n2, MCX_rand& ran) {
        float Rtotal = reflectcoeff(n1, n2);

        if (ran.rand01() > Rtotal) {
            (ipos.w == 0) ? (vec.x = -vec.x) : ((ipos.w == 1) ? (vec.y = -vec.y) : (vec.z = -vec.z)) ;
            return 0;
        } else {
            transmit(n1, n2);
            return 1;
        }
    }
    void rotatevector(float stheta, float ctheta, float sphi, float cphi) {
        if ( vec.z > -1.f + FLT_EPSILON && vec.z < 1.f - FLT_EPSILON ) {
            float tmp0 = 1.f - vec.z * vec.z;
            float tmp1 = stheta / sqrtf(tmp0);
            vec = (float4) {
                tmp1 * (vec.x * vec.z * cphi - vec.y * sphi) + vec.x* ctheta,
                     tmp1 * (vec.y * vec.z * cphi + vec.x * sphi) + vec.y* ctheta,
                     -tmp1* tmp0* cphi                          + vec.z* ctheta,
                     vec.w
            };
        } else {
            vec = (float4) {
                stheta* cphi, stheta* sphi, (vec.z > 0.f) ? ctheta : -ctheta, vec.w
            };
        }

        float tmp0 = 1.f / sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        vec.x *= tmp0;
        vec.y *= tmp0;
        vec.z *= tmp0;
    }
};
#pragma omp end declare target
/// MCX_clock class provides timing information
/** */
struct MCX_clock {
    std::chrono::system_clock::time_point starttime;
    MCX_clock() : starttime(std::chrono::system_clock::now()) {}
    double elapse() {
        std::chrono::duration<double> elapsetime = (std::chrono::system_clock::now() - starttime);
        return elapsetime.count() * 1000.;
    }
};
/// MCX_userio parses user JSON input and saves output to binary JSON files
/** */
struct MCX_userio {
    json cfg;
    MCX_userio(std::string finput) {
        if (finput == "cube60") {
            cfg = { {"Session", {{"ID", "cube60"}, {"Photons", 1000000}}}, {"Forward", {{"T0", 0.0}, {"T1", 5e-9}, {"Dt", 5e-9}}},
                {"Domain", {{"Media", { {{"mua", 0.0}, {"mus", 0.0}, {"g", 1.0}, {"n", 1.0}}, {{"mua", 0.005}, {"mus", 1.0}, {"g", 0.01}, {"n", 1.37}}}}, {"Dim", {60, 60, 60}}}},
                {"Optode", {{"Source", {{"Type", "pencil"}, {"Pos", {29.0, 29.0, 0.0}}, {"Dir", {0.0, 0.0, 1.0}}}}}},
                {"Shapes", {{"Grid", {{"Tag", 1}, {"Size", {60, 60, 60}}}}}}
            };
        } else {
            std::ifstream inputjson(finput);
            inputjson >> cfg;
        }
    }
    void save(MCX_volume<float>& outputvol, std::string outputfile = "output.bnii") {
        json bniifile = {
            {
                "NIFTIHeader", {
                    {"Dim", {cfg["Domain"]["Dim"][0], cfg["Domain"]["Dim"][1], cfg["Domain"]["Dim"][2]}}
                }
            },
            {
                "NIFTIData", {
                    {"_ArraySize_", {cfg["Domain"]["Dim"][0], cfg["Domain"]["Dim"][1], cfg["Domain"]["Dim"][2]}},
                    {"_ArrayType_", "single"}, {"_ArrayOrder_", "c"},
                    {"_ArrayData_", std::vector<float>(outputvol.buffer(), outputvol.buffer() + outputvol.count())}
                }
            }
        };
        std::ofstream outputdata(outputfile, std::ios::out | std::ios::binary);
        std::vector<uint8_t> output_vector;
        json::to_bjdata(bniifile, outputdata);
        outputdata.write((const char*)output_vector.data(), output_vector.size());
    }
};
/////////////////////////////////////////////////
/// \brief main function
/////////////////////////////////////////////////
int main(int argn, char* argv[]) {
    if (argn == 1) {
        std::cout << "format: umcx input.json" << std::endl;
        return 0;
    }

    MCX_userio io(argv[1]);
    MCX_volume<int> inputvol(io.cfg["Domain"]["Dim"][0], io.cfg["Domain"]["Dim"][1], io.cfg["Domain"]["Dim"][2], 1, 1);
    MCX_volume<float> outputvol(io.cfg["Domain"]["Dim"][0], io.cfg["Domain"]["Dim"][1], io.cfg["Domain"]["Dim"][2]);
    MCX_medium* prop = new MCX_medium[io.cfg["Domain"]["Media"].size()];

    for (uint32_t i = 0; i < io.cfg["Domain"]["Media"].size(); i++) {
        prop[i] = (MCX_medium) {
            io.cfg["Domain"]["Media"][i]["mua"], io.cfg["Domain"]["Media"][i]["mus"], io.cfg["Domain"]["Media"][i]["g"], io.cfg["Domain"]["Media"][i]["n"]
        };
    }

    double energyescape = 0.0;
    MCX_clock timer;
    uint64_t nphoton = io.cfg["Session"]["Photons"].get<uint64_t>();
    dim4 seeds = {(uint32_t)std::rand(), (uint32_t)std::rand(), (uint32_t)std::rand(), (uint32_t)std::rand()};  //< TODO: need to implement per-thread ran object
    float4 pos = {io.cfg["Optode"]["Source"]["Pos"][0], io.cfg["Optode"]["Source"]["Pos"][1], io.cfg["Optode"]["Source"]["Pos"][2], 1.f};
    float4 dir = {io.cfg["Optode"]["Source"]["Dir"][0], io.cfg["Optode"]["Source"]["Dir"][1], io.cfg["Optode"]["Source"]["Dir"][2], 0.f};
#ifdef GPU_OFFLOAD
    #pragma omp target teams distribute parallel for \
    map(to: inputvol) map(to: prop) map(tofrom: outputvol) reduction(+ : energyescape)
#else
    #pragma omp parallel for reduction(+ : energyescape)
#endif

    for (uint64_t i = 0; i < nphoton; i++) {
        MCX_rand ran(seeds.x ^ i, seeds.y | i, seeds.z ^ i, seeds.w | i);
        MCX_photon p(pos, dir);
        p.run(inputvol, outputvol, prop, ran);
        energyescape += p.pos.w;
    }

    printf("simulation completed %.6f ms, absorption fraction %.6f%%\n", timer.elapse(), (nphoton - energyescape) / nphoton * 100.);
    io.save(outputvol);
    delete [] prop;
    return 0;
}
