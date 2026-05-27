/*
Execute by: g++.exe (Rev3, Built by MSYS2 project) 14.2.0
g++ -std=c++20 -I -fPIC -shared -o ReRAM.dll ReRAM.cpp
*/

#include <iostream>
#include <cmath>
#include <numbers>
#include <random>
#include <fstream>

#include "CppUtilitiesFede.h"

using namespace std;

struct ReRAM_LAYER {
    float* W;
    int rows;
    int columns;
};

class ReRAM_LAYERs {
private:

public:


    ReRAM_LAYERs(vector<ReRAM_LAYER>& ReRAM_layers) {
        
        
    };
    bool Init() {
        return false;
    };
        
};


extern "C" {
    ReRAM_LAYERs* obj;
    void InitReRAMlayers(vector<ReRAM_LAYER>& ReRAMlayers) {
        
        obj = new ReRAM_LAYERs(ReRAMlayers);
    }
    
    void FreeSimulatorMemory() {
        // delete obj->randObj;
        delete obj;
    }
}

