/*
Execute by: g++.exe (Rev3, Built by MSYS2 project) 14.2.0
g++ -std=c++20 -I -fPIC -shared -o ReRAM.dll ReRAM.cpp
*/

#include <iostream>
#include <cmath>
#include <numbers>
#include <random>
#include <fstream>

#include "CppUtilitiesFede.hpp"

using namespace std;

struct RERAM_LAYER {
    unsigned ID_layer;

    int* pulsesPot;
    float* valsWpot;
    float* valsWpot_smooth;

    int* pulsesDep;
    float* valsWdep;
    float* valsWdep_smooth;

    // float* W;
    int rows;
    int columns;
};

class RERAM_PULSES {
private:
vector<RERAM_LAYER> ReRAM_layers;

public:


    RERAM_PULSES() {
        
        
    };
    bool Init() {
        return false;
    };
    void PushReRAMlayer(RERAM_LAYER ReRAMlayer) {
        ReRAM_layers.push_back(ReRAMlayer);
    }
    ~RERAM_PULSES() {

    }
};


extern "C" {
    RERAM_PULSES* obj;
    void InitReRAMlayers() {
        
        obj = new RERAM_PULSES();
    }
    void LoadReRAMlayer(RERAM_LAYER ReRAMlayer) {
        obj->PushReRAMlayer(ReRAMlayer);
    }
    
    void FreeSimulatorMemory() {
        delete obj;
    }
}

