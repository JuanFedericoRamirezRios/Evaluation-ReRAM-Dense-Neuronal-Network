/*
Execute by: g++.exe (Rev3, Built by MSYS2 project) 14.2.0
g++ -std=c++20 -I -fPIC -shared -o ReRAM.dll ReRAM.cpp
*/

#include <iostream>
#include <cmath>
#include <numbers>
#include <random>
#include <fstream>
// #include <vector>

#include "CppUtilitiesFede.hpp"

using namespace std;

// struct RERAM_VALS {
//     unsigned ID_layer;

//     int* pulsesPot; // (Ppot_max)
//     float* valsWpot; // (Ppot_max)
//     float* valsWpot_smooth; // (Ppot_max)

//     int* pulsesDep; // (Pdep_max)
//     float* valsWdep; // (Pdep_max)
//     float* valsWdep_smooth; // (Pdep_max)

//     // float* W;
//     int rows;
//     int columns;
// };

class RERAM_PULSES {
private:
    // vector<RERAM_VALS> ReRAM_layers;
    // unsigned ID_layer;

    int* pulsesPot; // (Ppot_max)
    float* valsWpot; // (Ppot_max)
    float* valsWpot_smooth; // (Ppot_max)

    int* pulsesDep; // (Pdep_max)
    float* valsWdep; // (Pdep_max)
    float* valsWdep_smooth; // (Pdep_max)

    float* W;
    int rows;
    int columns;

public:
    // RERAM_PULSES() {
        
        
    // };
    RERAM_PULSES(
        // unsigned ID_layer,

        int* pulsesPot, // (Ppot_max)
        float* valsWpot, // (Ppot_max)
        float* valsWpot_smooth, // (Ppot_max)

        int* pulsesDep, // (Pdep_max)
        float* valsWdep, // (Pdep_max)
        float* valsWdep_smooth, // (Pdep_max)

        int rows,
        int columns
    ) {
        
        
    };
    ~RERAM_PULSES() {

    };
    bool Init() {
        return false;
    };
    // void PushReRAMlayer(RERAM_VALS ReRAMlayer) {
    //     ReRAM_layers.push_back(ReRAMlayer);
    // };
};


extern "C" {
    vector<RERAM_PULSES*> layers;
    // RERAM_PULSES* ReRAMlayerObj;
    // void InitReRAMlayers() {
        
    //     ReRAMlayerObj = new RERAM_PULSES();
    // }
    void InitReRAMlayer(
        // unsigned ID_layer,

        int* pulsesPot, // (Ppot_max)
        float* valsWpot, // (Ppot_max)
        float* valsWpot_smooth, // (Ppot_max)

        int* pulsesDep, // (Pdep_max)
        float* valsWdep, // (Pdep_max)
        float* valsWdep_smooth, // (Pdep_max)

        int rows,
        int columns
    ) {
        
        RERAM_PULSES* ReRAMlayerObj = new RERAM_PULSES (
            // ID_layer,

            pulsesPot, // (Ppot_max)
            valsWpot, // (Ppot_max)
            valsWpot_smooth, // (Ppot_max)

            pulsesDep, // (Pdep_max)
            valsWdep, // (Pdep_max)
            valsWdep_smooth, // (Pdep_max)

            rows,
            columns
        );
        layers.push_back(ReRAMlayerObj);

    }
    void FreeMemory() {
        for(RERAM_PULSES* layer : layers) {
            delete(layer);
        }
        
        // delete ReRAMlayerObj;
    }
    void LearningReRAMlayer(float* array) {

    }

    // void LoadReRAMlayer(RERAM_VALS ReRAMlayer) {
    //     obj->PushReRAMlayer(ReRAMlayer);
    // }
}
    
    

