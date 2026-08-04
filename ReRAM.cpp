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

//     int* pulsesPot; // (Ppot_max+1)
//     float* valsWpot; // (Ppot_max+1)
//     float* valsWpot_smooth; // (Ppot_max+1)

//     int* pulsesDep; // (Pdep_max+1)
//     float* valsWdep; // (Pdep_max+1)
//     float* valsWdep_smooth; // (Pdep_max+1)

//     // float* W;
//     int rows;
//     int columns;
// };

class RERAM_LAYER {
private:
    // vector<RERAM_VALS> ReRAM_layers;
    // unsigned ID_layer;

    int totalPulses;

    int Ppot_max;
    float* valsWpot; // (Ppot_max): include experimental noise.
    float* valsWpot_smooth; // (Ppot_max): without experimental noise.

    int Pdep_max;
    float* valsWdep; // (Pdep_max)
    float* valsWdep_smooth; // (Pdep_max)

    float* W;
    // float** Wsmooth;
    int rows;
    int cols;

    float Wmin, Wmax;

public:
    // RERAM_PULSES() {
        
        
    // };
    RERAM_LAYER(
        // unsigned ID_layer,

        int Ppot_max,
        float* valsWpot, // (Ppot_max+1)
        float* valsWpot_smooth, // (Ppot_max+1)

        int Pdep_max,
        float* valsWdep, // (Pdep_max+1)
        float* valsWdep_smooth, // (Pdep_max+1)

        int rows,
        int cols
    ) {
        
        this->Ppot_max = Ppot_max;
        this->valsWpot = valsWpot;
        this->valsWpot_smooth = valsWpot_smooth;

        this->Pdep_max = Pdep_max;
        this->valsWdep = valsWdep;
        this->valsWdep_smooth = valsWdep_smooth;

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
        

        // cout << Wmin << " " << Wmax << endl;



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
        //     cout << valsWpot[n] << " ";
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
                    if(W >= newW) break;
                }
            }
        } else {
            for(int p = 0; p < Pdep_max; p++) {
                if(valsWdep[p] < Wcurrent) {
                    totalPulses++;
                    W = valsWdep[p];
                    // cout << W << " ";
                    if(W <= newW) break;
                }
            }
        }
        // cout << endl;

        /******* Check min and max W ********/
        if(W < Wmin) W = Wmin;
        if(W > Wmax) W = Wmax;        
    }
    float* ChangeWs(float* newWs) {
        for(int row = 0; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                ChangeW(W[row*cols + col], newWs[row*cols + col]);
                cout << newWs[row*cols + col] << " ";
            }
            cout << endl;
        }
        return W;
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
        // for(int row = 0; row < rows; row++) {
            // delete(W);
        // }
        delete(W);
    };
    
    // void PushReRAMlayer(RERAM_VALS ReRAMlayer) {
    //     ReRAM_layers.push_back(ReRAMlayer);
    // };
};


extern "C" {
    vector<RERAM_LAYER*> layers;
    // RERAM_LAYER* ReRAMlayerObj;
    // void InitReRAMlayers() {
        
    //     ReRAMlayerObj = new RERAM_PULSES();
    // }
    int InitReRAMlayer(
        // unsigned ID_layer,

        int Ppot_max,
        float* valsWpot, // (Ppot_max+1)
        float* valsWpot_smooth, // (Ppot_max+1)

        int Pdep_max,
        float* valsWdep, // (Pdep_max+1)
        float* valsWdep_smooth, // (Pdep_max+1)

        int rows,
        int cols
    ) {
        cout << "Init ReRAM layer: " << layers.size() << endl;
        RERAM_LAYER* ReRAMlayerObj = new RERAM_LAYER (
            // ID_layer,

            Ppot_max,
            valsWpot, // (Ppot_max+1)
            valsWpot_smooth, // (Ppot_max+1)

            Pdep_max,
            valsWdep, // (Pdep_max+1)
            valsWdep_smooth, // (Pdep_max+1)

            rows,
            cols
        );
        layers.push_back(ReRAMlayerObj);
        
        // cout << "Example depression:" << endl;
        // float _ = 0.0;
        // cout << "init: " << _ << endl;
        // ReRAMlayerObj->ChangeW(_, -1.0);
        // cout << "final: " << _ << endl;
        // cout << endl;


        return (int)(layers.size()-1); // Return the index of layer
        

    }
    float* ChangeWs(int numReRAMlayer, float* newW) {
        return layers[numReRAMlayer]->ChangeWs(newW);
        // layers[numReRAMlayer]->PrintReRAMlayer(numReRAMlayer);


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
        ;
    }
    
    void LearningReRAMlayer(int layerReRAM, float** newWsmooth) {
        /*
        newW: (rows, cols)
        potentiation: (rows, cols), if potentiation = false -> depression.
        */
        


    }

    // void LoadReRAMlayer(RERAM_VALS ReRAMlayer) {
    //     obj->PushReRAMlayer(ReRAMlayer);
    // }
}
    
    

