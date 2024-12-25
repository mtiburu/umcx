#include <iostream>
#include <fstream>
#include <string>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include "nlohmann/json.hpp"

using json = nlohmann::json;

struct float4 {
    float x = 0, y = 0, z = 0, w = 0;
    float4(float v) : x(v), y(v), z(v), w(v) {};
    float4(float x0, float y0, float z0, float w0) : x(x0), y(y0), z(z0), w(w0) {};
};

struct mcx_photon { // private, register-level
    float4 pos /*{x,y,z,w}*/, vec /*{vx,vy,vz,nscat}*/, len /*{pscat,t,pathlen,ndone}*/;
    void launch(float4 p0, float4 v0) {
        pos = p0;
        vec = v0;
        len = float4(0.f);
    };
};

template<class T>
class mcx_volume { // shared, read-only
    int nx = 0, ny = 0, nz = 0, nt = 0, dimxy = 0, dimxyz = 0, dimxyzt = 0;
    T* vol = nullptr;

  public:
    mcx_volume(int Nx, int Ny, int Nz, int Nt = 1) {
        nx = Nx;
        ny = Ny;
        nz = Nz;
        nt = Nt;
        dimxy = nx * ny;
        dimxyz = dimxy * ny;
        dimxyzt = dimxyz * nt;
        vol = new T[dimxyzt];
    };
    ~mcx_volume () {
        nx = ny = nz = nt = 0;
        delete [] vol;
        vol = nullptr;
    };
    void loadfromjnii (std::string fname) {
        std::ifstream inputjnii(fname);
        json jnii;
        inputjnii >> jnii;
        nx = jnii["NIFTIData"]["_ArraySize_"][0];
        ny = jnii["NIFTIData"]["_ArraySize_"][1];
        nz = jnii["NIFTIData"]["_ArraySize_"][2];
    };
    T& get(std::size_t x, std::size_t y, std::size_t z, std::size_t t = 1) const  {
        return vol[(t - 1) * dimxyzt + (z - 1) * dimxyz + (y - 1) * dimxy + x];
    };
    void set(T val, std::size_t x, std::size_t y, std::size_t z, std::size_t t = 1) const  {
        vol[(t - 1) * dimxyzt + (z - 1) * dimxyz + (y - 1) * dimxy + x] = val;
    };
};

int main(int argn, char* argv[]) {
    if (argn == 1) {
        std::cout << "format: umcx input.json" << std::endl;
        return 0;
    }

    std::ifstream inputjson(argv[1]);
    json cfg;
    inputjson >> cfg;

    mcx_volume<int> vol(cfg["Domain"]["Dim"][0], cfg["Domain"]["Dim"][1], cfg["Domain"]["Dim"][2]);
    return 0;
}
