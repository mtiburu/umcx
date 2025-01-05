//////////////////////////////////////////////////////////////////////////////////////////////////
///  \mainpage uMCX: readable, portable, hackable and massively-parallel photon simulator
///  \copyright Qianqian Fang <q.fang at neu.edu>, 2024-2025
///  \section sRationale Project Rationale
///       \li Must be readable, write clean C++11 code as short as possible without obscurity
///       \li Must be highly portable, support as many CPUs/GPUs and C++11 compilers as possible
///       \li Must use human-understandable user input/output, centered around JSON/binary JSON
///       \li Avoid using fancy C++ classes inside omp target region as OpenMP support is limited
///  \section sFormat Code Formatting
//        Please always run "make pretty" inside \c src before each commit, needing \c astyle
///  \section sLicense Open-source License
///       GPL v3 or later, see LICENSE.txt for details
//////////////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>
#include <math.h>
#include <chrono>
#include <set>
#include <map>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include "nlohmann/json.hpp"

#define ONE_OVER_C0          3.335640951981520e-12f
#define FLT_PI               3.1415926535897932385f
#define REFLECT_PHOTON(dir)  (vec.dir = -vec.dir, rvec.dir = -rvec.dir, pos.dir = nextafterf((int)(pos.dir + 0.5f), (vec.dir > 0.f) - (vec.dir < 0.f)), ipos.dir = (short)(pos.dir))
#define TRANSMIT_PHOTON(dir) (vec.dir = ((tmp0 = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z - vec.dir * vec.dir) < 1.f) ? sqrtf(1.f - tmp0) * ((vec.dir > 0.f) - (vec.dir < 0.f)) : 0.f)
#define JNUM(o, key1, key2)  (o[key1][key2].get<float>())
#define JVAL(o, key1, type)  (o[key1].get<type>())
#define JHAS(o, key1, type, default)   (o.contains(key1) ? (o[key1].get<type>()) : (default))

using json = nlohmann::ordered_json;
#pragma omp declare target
/// basic data type: float4 class, providing 4x float elements {x,y,z,w}, used for representing photon states
struct float4 {
    float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
    float4() {}
    float4(float x0, float y0, float z0, float w0) : x(x0), y(y0), z(z0), w(w0) {}
    float4& operator*=(float scale) {
        x *= scale, y *= scale, z *= scale;
        return *this;
    }
};
/// basic data type: dim4 class, representing array dimensions, with 4x uint32_t members {x,y,z,w}
struct dim4 {
    uint32_t x = 0u, y = 0u, z = 0u, w = 0u;
    dim4() {}
    dim4(uint32_t x0, uint32_t y0, uint32_t z0, uint32_t w0 = 0) : x(x0), y(y0), z(z0), w(w0) {}
};
/// basic data type: short4 class
struct short4 {
    int16_t x = 0, y = 0, z = 0, w = 0;
    short4() {}
    short4(int16_t x0, int16_t y0, int16_t z0, int16_t w0) : x(x0), y(y0), z(z0), w(w0) {}
};
/// Volumetric optical properties, 4 float members, mua (absorption coeff., 1/mm), mus (scattering coeff., 1/mm), g (anisotropy) and n (ref. coeff.)
struct MCX_medium {
    float mua = 0.f, mus = 0.f, g = 1.f, n = 1.f;
    MCX_medium() {}
    MCX_medium(float mua0, float mus0, float g0, float n0) : mua(mua0), mus(mus0), g(g0), n(n0) {}
};
/// Global simulation settings, all constants throughout the simulation
struct MCX_param {
    float tstart, tend, rtstep, unitinmm;
    int maxgate, isreflect, mediumnum, outputtype;
};
#pragma omp end declare target
/// MCX_volume class manages input and output volume
template<class T>
struct MCX_volume { // shared, read-only
    dim4 size;
    uint64_t dimxy = 0, dimxyz = 0, dimxyzt = 0;
    T* vol = nullptr;

    MCX_volume() {}
    MCX_volume(MCX_volume& v) {
        reshape(v.size.x, v.size.y, v.size.z, v.size.w);
        std::memcpy(vol, v.vol, sizeof(T)*dimxyzt);
    }
    MCX_volume(uint32_t Nx, uint32_t Ny, uint32_t Nz, uint32_t Nt = 1, T value = 0.0) {
        reshape(Nx, Ny, Nz, Nt, value);
    }
    void reshape(uint32_t Nx, uint32_t Ny, uint32_t Nz, uint32_t Nt = 1, T value = 0.0) {
        size = dim4(Nx, Ny, Nz, Nt);
        dimxy = Nx * Ny;
        dimxyz = dimxy * Ny;
        dimxyzt = dimxyz * Nt;
        delete [] vol;
        vol = new T[dimxyzt]();

        for (uint64_t i = 0; i < dimxyzt; i++) {
            vol[i] = value;
        }
    }
    ~MCX_volume () {
        delete [] vol;
    }
    int64_t index(short ix, short iy, short iz, int it = 0) { // when outside the volume, return -1, otherwise, return 1d index
        return !(ix < 0 || iy < 0 || iz < 0 || ix >= (short)size.x || iy >= (short)size.y || iz >= (short)size.z || it >= (int)size.w) ? (int)(it * dimxyz + iz * dimxy + iy * size.x + ix) : -1;
    }
    T& get(int64_t idx) const  { // must be inside the volume
        return vol[idx];
    }
    void add(T val, int64_t idx) const  {
        #pragma omp atomic
        vol[idx] += val;
    }
    void mask(T val, int64_t idx) const  {
        vol[idx] = (val > 0.) ? val : vol[idx];
    }
};
#pragma omp declare target
/// MCX_rand provides the xorshift128p random number generator
struct MCX_rand { // per thread
    uint64_t t[2];

    MCX_rand(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
        reseed(s0, s1, s2, s3);
    }
    void reseed(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
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
struct MCX_photon { // per thread
    float4 pos /*{x,y,z,w}*/, vec /*{vx,vy,vz,nscat}*/, rvec /*1/vx,1/vy,1/vz,unused*/, len /*{pscat,t,pathlen,p0}*/;
    short4 ipos /*{ix,iy,iz,flipdir}*/;
    int64_t lastvoxelidx;
    int mediaid;

    MCX_photon(const float4& p0, const float4& v0) { // constructor
        launch(p0, v0);
    }
    void launch(const float4& p0, const float4& v0) { // launch photon
        pos = p0;
        vec = v0;
        rvec = float4(1.f / v0.x, 1.f / v0.y, 1.f / v0.z, 1.f);
        len = float4(NAN, 0.f, 0.f, pos.w);
        ipos = short4((short)p0.x, (short)p0.y, (short)p0.z, -1);
        lastvoxelidx = -1;
        mediaid = 0;
    }
    template<const bool isreflect>              // main function to run a single photon from lunch to termination
    void run(MCX_volume<int>& invol, MCX_volume<float>& outvol, MCX_medium props[], MCX_rand& ran, const MCX_param& gcfg) {
        lastvoxelidx = outvol.index(ipos.x, ipos.y, ipos.z, 0);
        mediaid = invol.get(lastvoxelidx);
        len.x = ran.next_scat_len();

        while (sprint<isreflect>(invol, outvol, props, ran, gcfg) == 0) {
            scatter(props[mediaid], ran);
        }
    }
    template<const bool isreflect>              // run from one scattering site to the next, return 1 when terminate
    int sprint(MCX_volume<int>& invol, MCX_volume<float>& outvol, MCX_medium props[], MCX_rand& ran, const MCX_param& gcfg) {
        while (len.x > 0.f) {
            int64_t newvoxelid = step(invol, props[mediaid]);

            if (newvoxelid != lastvoxelidx) {   // only save when moving out of a voxel
                save(outvol, fminf(gcfg.maxgate - 1, (int)(floorf((len.y - gcfg.tstart) * gcfg.rtstep))), props[mediaid].mua, gcfg);

                if (len.y > gcfg.tend) {
                    return 1;                   // terminating photon due to exceeding maximum time gate
                }

                int newmediaid = ((newvoxelid >= 0) ? invol.get(newvoxelid) : 0);

                if (isreflect && gcfg.isreflect && props[mediaid].n != props[newmediaid].n) {
                    if (reflect(props[mediaid].n, props[newmediaid].n, ran, newvoxelid, newmediaid) && (newvoxelid < 0 || newmediaid == 0)) {
                        return 1;               // terminating photon due to transmitting to background at boundary
                    }
                } else if (newvoxelid < 0 || newmediaid == 0) {
                    return 1;                   // terminating photon due to continue moving to 0-valued voxel or out of domain
                }

                lastvoxelidx = newvoxelid;      // save last saving site
                mediaid = newmediaid;
            }
        }

        return 0;
    }
    int64_t step(MCX_volume<int>& invol, MCX_medium& prop) {
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
        pos = float4(pos.x + dist * vec.x, pos.y + dist * vec.y, pos.z + dist * vec.z, pos.w * expf(-prop.mua * dist));

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
    void save(MCX_volume<float>& outvol, int tshift, float mua, const MCX_param& gcfg) {
        outvol.add(gcfg.outputtype == 2 ? (len.w - pos.w) : (mua < FLT_EPSILON ? (len.w * len.z) : (len.w - pos.w) / mua), lastvoxelidx + tshift * outvol.dimxyz);
        len.w = pos.w;
        len.z = 0.f;
    }
    void scatter(MCX_medium& prop, MCX_rand& ran) {
        float tmp0, sphi, cphi, theta, stheta, ctheta;
        len.x = ran.next_scat_len();

        tmp0 = (2.f * FLT_PI) * ran.rand01(); //next arimuth angle
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
        rvec = float4(1.f / vec.x, 1.f / vec.y, 1.f / vec.z, 1.f);
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

        vec *= tmp0;
        (ipos.w == 0) ? TRANSMIT_PHOTON(x) : ((ipos.w == 1) ? TRANSMIT_PHOTON(y) : TRANSMIT_PHOTON(z));
        tmp0 = 1.f / sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        vec *= tmp0;
    }
    int reflect(float n1, float n2, MCX_rand& ran, int64_t& newvoxelid, int& newmediaid) {
        float Rtotal = reflectcoeff(n1, n2);

        if (Rtotal < 1.f && ran.rand01() > Rtotal) {
            transmit(n1, n2);
            return 1;
        } else {
            (ipos.w == 0) ? REFLECT_PHOTON(x) : ((ipos.w == 1) ? REFLECT_PHOTON(y) : REFLECT_PHOTON(z));
            newvoxelid = lastvoxelidx;
            newmediaid = mediaid;
            return 0;
        }
    }
    void rotatevector(float stheta, float ctheta, float sphi, float cphi) {
        if ( vec.z > -1.f + FLT_EPSILON && vec.z < 1.f - FLT_EPSILON ) {
            float tmp0 = 1.f - vec.z * vec.z;
            float tmp1 = stheta / sqrtf(tmp0);
            vec = float4(tmp1 * (vec.x * vec.z * cphi - vec.y * sphi) + vec.x * ctheta,
                         tmp1 * (vec.y * vec.z * cphi + vec.x * sphi) + vec.y * ctheta,
                         -tmp1 * tmp0 * cphi                          + vec.z * ctheta,
                         vec.w);
        } else {
            vec = float4(stheta * cphi, stheta * sphi, (vec.z > 0.f) ? ctheta : -ctheta, vec.w);
        }

        float tmp0 = 1.f / sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        vec *= tmp0;
    }
    void sincosf(float ang, float* sine, float* cosine) {
        *sine = sinf(ang);
        *cosine = cosf(ang);
    }
};
#pragma omp end declare target
/// MCX_clock class provides run-time information
struct MCX_clock {
    std::chrono::system_clock::time_point starttime;
    MCX_clock() : starttime(std::chrono::system_clock::now()) {}
    double elapse() {
        std::chrono::duration<double> elapsetime = (std::chrono::system_clock::now() - starttime);
        return elapsetime.count() * 1000.;
    }
};
const json MCX_benchmarks = {"cube60", "cube60b", "cubesph60b", "sphshells", "spherebox", "skinvessel"};
enum MCX_benchmarkid {bm_cube60, bm_cube60b, bm_cubesph60b, bm_sphshells, bm_spherebox, bm_skinvessel};
const std::string MCX_outputtype = "xfe";
/// MCX_userio parses user JSON input and saves output to binary JSON files
struct MCX_userio {
    json cfg;
    MCX_volume<int> domain;
    const std::map<std::set<std::string>, std::string> cmdflags = {{{"-n", "--photon"}, "/Session/Photons"}, {{"-b", "--reflect"}, "/Session/DoMismatch"},
        {{"-u", "--unitinmm"}, "/Domain/LengthUnit"}, {{"-U", "--normalize"}, "/Session/DoNormalize"}, {{"-E", "--seed"}, "/Session/RNGSeed"}, {{"-O", "--outputtype"}, "/Session/OutputType"},
        {{"-d", "--savedet"}, "/Session/DoPartialPath"}, {{"-t", "--thread"}, "/Session/ThreadNum"}, {{"-T", "--blocksize"}, "/Session/BlockSize"}, {{"-G", "--gpuid"}, "/Session/DeviceID"}
    };
    const std::map<std::string, std::function<int(float4 p, json obj)>> shapeparser = {
        {"Sphere", [](float4 p, json obj) -> int { return ((p.x - JNUM(obj, "O", 0)) * (p.x - JNUM(obj, "O", 0)) + (p.y - JNUM(obj, "O", 1)) * (p.y - JNUM(obj, "O", 1)) + (p.z - JNUM(obj, "O", 2)) * (p.z - JNUM(obj, "O", 2)) < (JVAL(obj, "R", float) * JVAL(obj, "R", float))) ? JVAL(obj, "Tag", int) : std::numeric_limits<int>::quiet_NaN(); }},
        {"Box", [](float4 p, json obj) -> int { return !(p.x < JNUM(obj, "O", 0) || p.y < JNUM(obj, "O", 1) || p.z < JNUM(obj, "O", 2) || p.x > JNUM(obj, "O", 0) + JNUM(obj, "Sip.ze", 0) || p.x > JNUM(obj, "O", 1) + JNUM(obj, "Sip.ze", 1) || p.x > JNUM(obj, "O", 2) + JNUM(obj, "Sip.ze", 2)) ? JVAL(obj, "Tag", int) : std::numeric_limits<int>::quiet_NaN(); }},
        {"XLayers", [](float4 p, json obj) -> int { return (p.x >= JVAL(obj, 0, float) && p.x <= JVAL(obj, 1, float)) ? obj[2].get<int>() : std::numeric_limits<int>::quiet_NaN(); }},
        {"YLayers", [](float4 p, json obj) -> int { return (p.y >= JVAL(obj, 0, float) && p.y <= JVAL(obj, 1, float)) ? obj[2].get<int>() : std::numeric_limits<int>::quiet_NaN(); }},
        {"ZLayers", [](float4 p, json obj) -> int { return (p.z >= JVAL(obj, 0, float) && p.z <= JVAL(obj, 1, float)) ? obj[2].get<int>() : std::numeric_limits<int>::quiet_NaN(); }},
        {
            "Cylinder", [](float4 p, json obj) -> int {
                const float d0 = (JNUM(obj, "C1", 0) - JNUM(obj, "C0", 0)) * (JNUM(obj, "C1", 0) - JNUM(obj, "C0", 0)) + (JNUM(obj, "C1", 1) - JNUM(obj, "C0", 1)) * (JNUM(obj, "C1", 1) - JNUM(obj, "C0", 1)) + (JNUM(obj, "C1", 2) - JNUM(obj, "C0", 2)) * (JNUM(obj, "C1", 2) - JNUM(obj, "C0", 2));
                const float4 v0((JNUM(obj, "C1", 0) - JNUM(obj, "C0", 0)) / sqrtf(d0), (JNUM(obj, "C1", 1) - JNUM(obj, "C0", 1)) / sqrtf(d0), (JNUM(obj, "C1", 2) - JNUM(obj, "C0", 2)) / sqrtf(d0), 0.f);
                float4 p0((p.x - JNUM(obj, "C0", 0)), (p.y - JNUM(obj, "C0", 1)), (p.z - JNUM(obj, "C0", 2)), d0);
                float d = v0.x * p0.x + v0.y * p0.y + v0.z * p0.z;
                return (d <= p0.w && d >= 0.f && p0.x * p0.x + p0.y * p0.y + p0.z * p0.z - d * d <= JVAL(obj, "R", float) * JVAL(obj, "R", float)) ? JVAL(obj, "Tag", int) : std::numeric_limits<int>::quiet_NaN();
            }
        }
    };
    MCX_userio(char* argv[], int argn = 1) {   // parsing command line
        std::string finput = argv[1];

        if (finput[0] == '-') {  // format 1: umcx -flag1 jsonvalue1 -flag2 jsonvalue2 --longflag3 jsonvalue3 ....
            int i = 1;

            while (i < argn) {
                std::string arg(argv[i++]);

                if (arg == "-f" || arg == "--input") {
                    loadfromfile(argv[i++]);
                } else if (arg == "--bench") {
                    benchmark(argv[i++]);
                } else if (arg == "-j" || arg == "--json") {
                    cfg.update(json::parse(argv[i++]), true);
                } else if (arg[0] == '-') {
                    for ( const auto& opts : cmdflags ) {
                        if (opts.first.find(arg) != opts.first.end()) {
                            cfg[json::json_pointer(opts.second)] = json::parse(argv[i++]);
                            break;
                        }
                    }
                }
            }
        } else if (finput.find(".") == std::string::npos) { // format 2: umcx benchmarkname
            benchmark(finput);
        } else {                                            // format 3: umcx input.json
            loadfromfile(finput);
        }

        initdomain();
    }
    void initdomain() {
        domain.reshape(cfg["Domain"]["Dim"][0], cfg["Domain"]["Dim"][1], cfg["Domain"]["Dim"][2]);

        if (cfg.contains("Shapes")) {
            if (!cfg["Shapes"].contains("_ArraySize_")) {
                json shapes = cfg["Shapes"].is_array() ? cfg["Shapes"][0] : cfg["Shapes"];

                if (shapes.contains("Grid")) {
                    domain.reshape(shapes["Grid"]["Size"][0], shapes["Grid"]["Size"][1], shapes["Grid"]["Size"][2], 1, shapes["Grid"]["Tag"]);

                    if ((cfg["Shapes"].is_array() && cfg["Shapes"].size() == 1) || cfg["Shapes"].is_object()) {
                        return;
                    }
                }

                for (const auto& obj : cfg["Shapes"]) {
                    #pragma omp parallel for collapse(2)

                    for (uint32_t x = 0; x < domain.size.x; x++)
                        for (uint32_t y = 0; y < domain.size.y; y++)
                            for (uint32_t z = 0; z < domain.size.z; z++) {
                                if (shapeparser.find(obj.begin().key()) != shapeparser.end()) {
                                    int64_t idx = domain.index(x, y, z);

                                    if (obj.begin().key().find("Layers") == 1) {
                                        for (auto layer : obj.front().items()) {
                                            int label = shapeparser.at(obj.begin().key())(float4(x + 0.5f, y + 0.5f, z + 0.5f, 0.f), layer.value());
                                            domain.mask((!std::isnan(label) ? label : 0), idx);
                                        }
                                    } else {
                                        int label = shapeparser.at(obj.begin().key())(float4(x + 0.5f, y + 0.5f, z + 0.5f, 0.f), obj.front());
                                        domain.mask((!std::isnan(label) ? label : 0), idx);
                                    }
                                }
                            }
                }
            } else {
                domain.reshape(cfg["Shapes"]["_ArraySize_"][0], cfg["Shapes"]["_ArraySize_"][1], cfg["Shapes"]["_ArraySize_"][2]);
            }
        }
    }
    void benchmark(std::string benchname) {
        MCX_benchmarkid bmid = (MCX_benchmarkid)std::distance(MCX_benchmarks.begin(), std::find(MCX_benchmarks.begin(), MCX_benchmarks.end(), benchname));
        cfg = { {"Session", {{"ID", "cube60"}, {"Photons", 1000000}, {"RNGSeed", 1648335518}}}, {"Forward", {{"T0", 0.0}, {"T1", 5e-9}, {"Dt", 5e-9}}},
            {"Domain", {{"Media", {{{"mua", 0.0}, {"mus", 0.0}, {"g", 1.0}, {"n", 1.0}}, {{"mua", 0.005}, {"mus", 1.0}, {"g", 0.01}, {"n", 1.37}}, {{"mua", 0.002}, {"mus", 5.0}, {"g", 0.9}, {"n", 1.0}}}}, {"Dim", {60, 60, 60}}}},
            {"Optode", {{"Source", {{"Type", "pencil"}, {"Pos", {29.0, 29.0, 0.0}}, {"Dir", {0.0, 0.0, 1.0}}}}}}
        };
        cfg["Shapes"] = R"([{"Grid": {"Tag": 1, "Size": [60, 60, 60]}}])"_json;
        cfg["Session"]["DoMismatch"] = !((int)(bmid == bm_cube60 || bmid == bm_skinvessel || bmid == bm_spherebox));

        if (bmid == bm_cubesph60b) {
            cfg["Shapes"] = R"([{"Grid": {"Tag": 1, "Size": [60, 60, 60]}}, {"Sphere": {"O": [30, 30, 30], "R": 15, "Tag": 2}}])"_json;
        } else if (bmid == bm_skinvessel) {
            cfg["Shapes"] = R"([{"Grid": {"Size": [200, 200, 200], "Tag": 1}}, {"ZLayers": [[1, 20, 1], [21, 32, 4], [33, 200, 3]]}, {"Cylinder": {"Tag": 2, "C0": [0, 100.5, 100.5], "C1": [200, 100.5, 100.5], "R": 20}}])"_json;
            cfg["Forward"] = {{"T0", 0.0}, {"T1", 5e-8}, {"Dt", 5e-8}};
            cfg["Optode"]["Source"] = {{"Type", "disk"}, {"Pos", {100, 100, 20}}, {"Dir", {0, 0, 1}}, {"Param1", {60, 0, 0, 0}}};
            cfg["Domain"]["LengthUnit"] = 0.005;
            cfg["Domain"]["Media"] = {{{"mua", 1e-5}, {"mus", 0.0}, {"g", 1.0}, {"n", 1.37}}, {{"mua", 3.564e-05}, {"mus", 1.0}, {"g", 1.0}, {"n", 1.37}}, {{"mua", 23.05426549}, {"mus", 9.398496241}, {"g", 0.9}, {"n", 1.37}},
                {{"mua", 0.04584957865}, {"mus", 35.65405549}, {"g", 0.9}, {"n", 1.37}}, {{"mua", 1.657237447}, {"mus", 37.59398496}, {"g", 0.9}, {"n", 1.37}}
            };
        } else if (bmid == bm_sphshells) {
            cfg["Shapes"] = R"([{"Grid": {"Size": [60, 60, 60], "Tag": 1}}, {"Sphere": {"O": [30, 30, 30], "R": 25, "Tag": 2}}, {"Sphere": {"O": [30, 30, 30], "R": 23, "Tag": 3}}, {"Sphere": {"O": [30, 30, 30], "R": 10, "Tag": 4}}])"_json;
            cfg["Domain"]["Media"] = {{{"mua", 0.0}, {"mus", 0.0}, {"g", 1.0}, {"n", 1.0}}, {{"mua", 0.02}, {"mus", 7.0}, {"g", 0.89}, {"n", 1.37}}, {{"mua", 0.004}, {"mus", 0.09}, {"g", 0.89}, {"n", 1.37}},
                {{"mua", 0.02}, {"mus", 9.0}, {"g", 0.89}, {"n", 1.37}}, {{"mua", 0.05}, {"mus", 0.0}, {"g", 1.0}, {"n", 1.37}}
            };
        } else if (bmid == bm_spherebox) {
            cfg["Forward"]["Dt"] = 1e-10;
            cfg["Domain"]["Media"] = {{{"mua", 0.0}, {"mus", 0.0}, {"g", 1.0}, {"n", 1.0}}, {{"mua", 0.002}, {"mus", 1.0}, {"g", 0.01}, {"n", 1.37}}, {{"mua", 0.005}, {"mus", 5.0}, {"g", 0.9}, {"n", 1.37}}};
            cfg["Shapes"] = R"([{"Grid": {"Tag": 1, "Size": [60, 60, 60]}}, {"Sphere": {"O": [30, 30, 30], "R": 10, "Tag": 2}}])"_json;
        }
    }
    void loadfromfile(std::string finput) {
        std::ifstream inputjson(finput);
        inputjson >> cfg;
    }
    template<class T>
    void save(MCX_volume<T>& outputvol, std::string outputfile = "output.bnii") {
        json bniifile = {
            { "NIFTIHeader", {{"Dim", {outputvol.size.x, outputvol.size.y, outputvol.size.z, outputvol.size.w}}}},
            {
                "NIFTIData", {{"_ArraySize_", {outputvol.size.x, outputvol.size.y, outputvol.size.z, outputvol.size.w}},
                    {"_ArrayType_", "single"}, {"_ArrayOrder_", "c"},
                    {"_ArrayData_", std::vector<T>(outputvol.vol, outputvol.vol + outputvol.dimxyzt)}
                }
            }
        };
        bniifile["NIFTIData"]["_ArrayType_"] = ((std::string(typeid(T).name()) == "f") ? "single" : "int32");
        std::ofstream outputdata(outputfile, std::ios::out | std::ios::binary);
        std::vector<uint8_t> output_vector;
        json::to_bjdata(bniifile, outputdata, true, true);
        outputdata.write((const char*)output_vector.data(), output_vector.size());
    }
};
/////////////////////////////////////////////////
/// \brief main function
/////////////////////////////////////////////////
int main(int argn, char* argv[]) {
    if (argn == 1) {
        std::cout << "Format: umcx -flag1 'jsonvalue1' -flag2 'jsonvalue2' ...\n\t\tor\n\tumcx inputjson.json\n\tumcx benchmarkname\n\nFlags:\n\t-f/--input\tinput json file\n\t-n/--photon\tphoton number\n\t--bench\t\tbenchmark name" << std::endl;
        std::cout << "\t-u/--unitinmm\tvoxel size in mm [1]\n\t-E/--seed\tRNG seed []\n\t-O/--outputtype\toutput type (x/f/e)\n\t-G/--gpuid\tdevice ID (1,2,...)\n\nAvailable benchmarks include: " << MCX_benchmarks.dump(8) << std::endl;
        return 0;
    }

    MCX_userio io(argv, argn);
    const MCX_param gcfg = {
        /*.tstart*/ JNUM(io.cfg, "Forward", "T0"), /*.tend*/ JNUM(io.cfg, "Forward", "T1"), /*.rtstep*/ 1.f / JNUM(io.cfg, "Forward", "Dt"), /*.unitinmm*/ (io.cfg["Domain"].contains("LengthUnit") ? JNUM(io.cfg, "Domain", "LengthUnit") : 1.f),
        /*.maxgate*/ (int)((JNUM(io.cfg, "Forward", "T1") - JNUM(io.cfg, "Forward", "T0")) / JNUM(io.cfg, "Forward", "Dt") + 0.5f),
        /*.isreflect*/ (io.cfg["Session"].contains("DoMismatch") ? io.cfg["Session"]["DoMismatch"].get<int>() : 0),
        /*.mediumnum*/ (int)io.cfg["Domain"]["Media"].size(), /*.outputtype*/ (int)MCX_outputtype.find(JHAS(io.cfg["Session"], "OutputType", std::string, "f")[0])
    };
    MCX_volume<int> inputvol = io.domain;
    MCX_volume<float> outputvol(io.cfg["Domain"]["Dim"][0].get<int>(), io.cfg["Domain"]["Dim"][1].get<int>(), io.cfg["Domain"]["Dim"][2].get<int>(), gcfg.maxgate);
    MCX_medium* prop = new MCX_medium[gcfg.mediumnum];

    for (int i = 0; i < gcfg.mediumnum; i++) {
        prop[i] = MCX_medium(JNUM(io.cfg["Domain"]["Media"], i, "mua") * gcfg.unitinmm, JNUM(io.cfg["Domain"]["Media"], i, "mus") * gcfg.unitinmm, io.cfg["Domain"]["Media"][i]["g"], io.cfg["Domain"]["Media"][i]["n"]);
    }

    double energyescape = 0.0;
    MCX_clock timer;
    std::srand(!(io.cfg["Session"].contains("RNGSeed")) ? 1648335518 : (io.cfg["Session"]["RNGSeed"].get<int>() > 0 ? io.cfg["Session"]["RNGSeed"].get<int>() : std::time(0)));
    const uint64_t nphoton = io.cfg["Session"]["Photons"].get<uint64_t>();
    const dim4 seeds = {(uint32_t)std::rand(), (uint32_t)std::rand(), (uint32_t)std::rand(), (uint32_t)std::rand()};  //< TODO: need to implement per-thread ran object
    const float4 pos = {io.cfg["Optode"]["Source"]["Pos"][0].get<float>(), io.cfg["Optode"]["Source"]["Pos"][1].get<float>(), io.cfg["Optode"]["Source"]["Pos"][2].get<float>(), 1.f};
    const float4 dir = {io.cfg["Optode"]["Source"]["Dir"][0].get<float>(), io.cfg["Optode"]["Source"]["Dir"][1].get<float>(), io.cfg["Optode"]["Source"]["Dir"][2].get<float>(), 0.f};
    MCX_rand ran(seeds.x, seeds.y, seeds.z, seeds.w);
    MCX_photon p(pos, dir);
#ifdef GPU_OFFLOAD
    const int deviceid = JHAS(io.cfg["Session"], "DeviceID", int, 1) - 1, gridsize = JHAS(io.cfg["Session"], "ThreadNum", int, 10000) / JHAS(io.cfg["Session"], "BlockSize", int, 64);
#ifdef _LIBGOMP_OMP_LOCK_DEFINED
    const int blocksize = JHAS(io.cfg["Session"], "BlockSize", int, 64) / 32;  // gcc nvptx offloading uses {32,teams_thread_limit,1} as blockdim
#else
    const int blocksize = JHAS(io.cfg["Session"], "BlockSize", int, 64); // nvc uses {num_teams,1,1} as griddim and {teams_thread_limit,1,1} as blockdim
#endif
    #pragma omp target teams distribute parallel for num_teams(gridsize) thread_limit(blocksize) device(deviceid) \
    map(to: pos) map(to: dir) map(to: seeds) map(to: gcfg) map(to: prop[0:gcfg.mediumnum]) reduction(+ : energyescape) firstprivate(ran, p) \
    map(to: inputvol) map(to: inputvol.vol[0:inputvol.dimxyzt]) map(tofrom: outputvol) map(tofrom: outputvol.vol[0:outputvol.dimxyzt])
#else
    #pragma omp parallel for reduction(+ : energyescape) firstprivate(ran, p)
#endif

    for (uint64_t i = 0; i < nphoton; i++) {
        ran.reseed(seeds.x ^ i, seeds.y | i, seeds.z ^ i, seeds.w | i);
        p.launch(pos, dir);

        if (gcfg.isreflect) {
            p.run<true>(inputvol, outputvol, prop, ran, gcfg);
        } else {
            p.run<false>(inputvol, outputvol, prop, ran, gcfg);
        }

        energyescape += p.pos.w;
    }

    printf("simulated energy %.2f, speed %.2f photon/ms, duration %.6f ms, absorbed %.6f%%\n", (double)nphoton, nphoton / timer.elapse(), timer.elapse(), (nphoton - energyescape) / nphoton * 100.);
    io.save<float>(outputvol);
    delete [] prop;
    return 0;
}
