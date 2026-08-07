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
using namespace UTILS_FEDE;


class RERAM_LAYER {
private:
    int totalPulses;

    float* W;
    int rows;
    int cols;

    float Wmin, Wmax;
public:
    int Ppot_max;
    float* valsWpot; // (Ppot_max): include experimental noise.
    float* valsWpot_smooth; // (Ppot_max): without experimental noise.

    int Pdep_max;
    float* valsWdep; // (Pdep_max)
    float* valsWdep_smooth; // (Pdep_max)

    RERAM_LAYER(
        // unsigned ID_layer,

        int Ppot_max,
        const float* valsWpot, // (Ppot_max)
        const float* valsWpot_smooth, // (Ppot_max)

        int Pdep_max,
        const float* valsWdep, // (Pdep_max)
        const float* valsWdep_smooth, // (Pdep_max)

        int rows,
        int cols
    ) {
        
        this->Ppot_max = Ppot_max;
        this->valsWpot = Copy1DfloatArray(valsWpot, Ppot_max);
        this->valsWpot_smooth = Copy1DfloatArray(valsWpot_smooth, Ppot_max);

        this->Pdep_max = Pdep_max;
        this->valsWdep = Copy1DfloatArray(valsWdep, Pdep_max);
        this->valsWdep_smooth = Copy1DfloatArray(valsWdep_smooth, Pdep_max);

        this->rows = rows;
        this->cols = cols;

        Wmin = 0.0f;
        Wmax = 0.0f;
        for(int n = 0; n < Ppot_max; n++) {
            if(valsWpot[n] < Wmin) Wmin = valsWpot[n];
            if(valsWpot[n] > Wmax) Wmax = valsWpot[n];
        }
        for(int n = 0; n < Pdep_max; n++) {
            if(valsWdep[n] < Wmin) Wmin = valsWdep[n];
            if(valsWdep[n] > Wmax) Wmax = valsWdep[n];
        }

        totalPulses = 0;
        
        W = new float[rows*cols];
        
        for(int row = 0; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                W[row*cols + col] = valsWpot[0];
                // cout << W[row*cols + col] << " ";
            }
            // cout << endl;
        }

        // cout << endl;
        // for(int n = 0; n < Ppot_max; n++) {
        //     cout << this->valsWpot[n] << " ";
        // }
        // cout << endl;

    };
    
    void ChangeW(float& W, float newW) {
        float Wcurrent = W;      
        if(W < newW) {
            for(int p = 0; p < Ppot_max; p++) {
                if(valsWpot[p] > Wcurrent) {
                    totalPulses++;
                    W = valsWpot[p];
                    // cout << W << " ";
                    if(W >= newW) {
                        cout << valsWpot[p] << " ";
                        break;
                    }
                }
            }
        } else {
            for(int p = 0; p < Pdep_max; p++) {
                if(valsWdep[p] < Wcurrent) {
                    totalPulses++;
                    W = valsWdep[p];
                    // cout << W << " ";
                    if(W <= newW) {
                        cout << valsWpot[p] << " ";
                        break;
                    }
                }
            }
        }
        

        // cout << endl;

        /******* Check min and max W ********/
        if(W < Wmin) W = Wmin;
        if(W > Wmax) W = Wmax;

        
    }
    float* ChangeWs(const float* newWs) {
        cout << endl;
        cout << Ppot_max << endl;
        for(int n = 0; n < Ppot_max; n++) {
            cout << valsWpot[n] << " ";
        }
        cout << endl;


        for(int row = 0; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                ChangeW(W[row*cols + col], newWs[row*cols + col]);
            }
            cout << endl;
        }

        return Copy1DfloatArray(W, rows*cols);
    };
    void PrintReRAMlayer(int numReRAMlayer) {
        cout << "ReRAM layer " << numReRAMlayer << ":" << endl;
        for(int row = 0; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                cout << W[row*cols + col] << " ";
            }
            cout << endl;
        }
        cout << endl;
    };
    ~RERAM_LAYER() {
        
        delete(W);
        delete(valsWpot); delete(valsWpot_smooth);
        delete(valsWdep); delete(valsWdep_smooth);

    };
};


extern "C" {
    vector<RERAM_LAYER*> layers;
    
    int InitReRAMlayer(
        int Ppot_max,
        float* valsWpot, // (Ppot_max)
        float* valsWpot_smooth, // (Ppot_max)

        int Pdep_max,
        float* valsWdep, // (Pdep_max)
        float* valsWdep_smooth, // (Pdep_max)

        int rows,
        int cols
    ) {
        cout << "Init ReRAM layer: " << layers.size() << endl;
        RERAM_LAYER* ReRAMlayerObj = new RERAM_LAYER (
            // ID_layer,

            Ppot_max,
            valsWpot, // (Ppot_max)
            valsWpot_smooth, // (Ppot_max)

            Pdep_max,
            valsWdep, // (Pdep_max)
            valsWdep_smooth, // (Pdep_max)

            rows,
            cols
        );
        layers.push_back(ReRAMlayerObj);

        int idLayer = (int)(layers.size()-1);
        
        // cout << "Example depression:" << endl;
        // float _ = -2.44949;
        // cout << "init: " << _ << endl;
        // layers[idLayer]->ChangeW(_, -1.52783);
        // cout << "final: " << _ << endl;
        // cout << endl;

        cout << endl;
        cout << layers[idLayer]->valsWpot[0] << endl;
        cout << endl;


        return idLayer; // Return the index of layer
        

    }
    float* ChangeWs(int idLayer, float* newW) {
        cout << endl;
        cout << layers[idLayer]->valsWpot[0] << endl;
        cout << endl;
    
        return layers[idLayer]->ChangeWs(newW);
    }
    void PrintReRAMlayer(int numReRAMlayer) {
            layers[numReRAMlayer]->PrintReRAMlayer(numReRAMlayer);
    }
    void FreeMemory() {
        
        for(RERAM_LAYER* layer : layers) {
            delete layer;
        }
        
        layers.clear();

        cout << "Size of vector layers after FreeMemory = " << layers.size() << endl;
    }
    
    void LearningReRAMlayer(int layerReRAM, float* newWsmooth) {
        /*
        newW: (rows, cols)
        potentiation: (rows, cols), if potentiation = false -> depression.
        */
        


    }

    // void LoadReRAMlayer(RERAM_VALS ReRAMlayer) {
    //     obj->PushReRAMlayer(ReRAMlayer);
    // }
}
    
    

