#include <iostream>
#include <iomanip>
using namespace std;

#define N 17
#define M 3

class Dataset {
public:
    double X[N][M];
    double Y[N];

    Dataset() {
        double x1[N] = {41.9,43.4,43.9,44.5,47.3,47.5,47.9,50.2,52.8,53.2,56.7,57.0,63.5,65.3,71.1,77.0,77.8};
        double x2[N] = {29.1,29.3,29.5,29.7,29.9,30.3,30.5,30.7,30.8,30.9,31.5,31.7,31.9,32.0,32.1,32.5,32.9};
        double y[N]  = {251.3,251.3,248.3,267.5,273.0,276.5,270.3,274.9,285.0,290.0,297.0,302.5,304.5,309.3,321.7,330.7,349.0};

        for(int i=0; i<N; i++) {
            X[i][0] = 1;      // B0
            X[i][1] = x1[i];  // x1
            X[i][2] = x2[i];  // x2
            Y[i] = y[i];
        }
    }
};

class LinearRegression {
private:
    double B[M];

public:

    void transpose(double A[N][M], double AT[M][N]) {
        for(int i=0;i<N;i++)
            for(int j=0;j<M;j++)
                AT[j][i] = A[i][j];
    }

    void multiply(double A[M][N], double Bm[N][M], double R[M][M]) {
        for(int i=0;i<M;i++) {
            for(int j=0;j<M;j++) {
                R[i][j] = 0;
                for(int k=0;k<N;k++)
                    R[i][j] += A[i][k] * Bm[k][j];
            }
        }
    }

    void multiplyVec(double A[M][N], double Y[N], double R[M]) {
        for(int i=0;i<M;i++) {
            R[i] = 0;
            for(int j=0;j<N;j++)
                R[i] += A[i][j] * Y[j];
        }
    }

    bool inverse3x3(double A[M][M], double inv[M][M]) {
        double det =
            A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) -
            A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) +
            A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);

        if(det == 0) return false;

        double invDet = 1.0/det;

        inv[0][0] =  (A[1][1]*A[2][2] - A[1][2]*A[2][1])*invDet;
        inv[0][1] = -(A[0][1]*A[2][2] - A[0][2]*A[2][1])*invDet;
        inv[0][2] =  (A[0][1]*A[1][2] - A[0][2]*A[1][1])*invDet;

        inv[1][0] = -(A[1][0]*A[2][2] - A[1][2]*A[2][0])*invDet;
        inv[1][1] =  (A[0][0]*A[2][2] - A[0][2]*A[2][0])*invDet;
        inv[1][2] = -(A[0][0]*A[1][2] - A[0][2]*A[1][0])*invDet;

        inv[2][0] =  (A[1][0]*A[2][1] - A[1][1]*A[2][0])*invDet;
        inv[2][1] = -(A[0][0]*A[2][1] - A[0][1]*A[2][0])*invDet;
        inv[2][2] =  (A[0][0]*A[1][1] - A[0][1]*A[1][0])*invDet;

        return true;
    }

    void fit(Dataset &data) {
        double XT[M][N];
        double XTX[M][M];
        double XTX_inv[M][M];
        double XTY[M];

        transpose(data.X, XT);
        multiply(XT, data.X, XTX);
        inverse3x3(XTX, XTX_inv);
        multiplyVec(XT, data.Y, XTY);

        for(int i=0;i<M;i++) {
            B[i] = 0;
            for(int j=0;j<M;j++)
                B[i] += XTX_inv[i][j] * XTY[j];
        }
    }

    double predict(double x1, double x2) {
        return B[0] + B[1]*x1 + B[2]*x2;
    }

    void printResults() {
        cout << fixed << setprecision(4);

        cout << "========================================\n";
        cout << "   REGRESION LINEAL MULTIPLE (LSR)\n";
        cout << "   DataSet: 17 Chemical Experiments\n";
        cout << "========================================\n\n";

        cout << "Parametros del modelo:\n";
        cout << "B0 (Intercepto) = " << B[0] << endl;
        cout << "B1 (Factor x1)  = " << B[1] << endl;
        cout << "B2 (Factor x2)  = " << B[2] << endl;

        cout << "\nEcuacion de Regresion:\n";
        cout << "Yield (y) = "
             << B[0] << " + ("
             << B[1] << " * x1) + ("
             << B[2] << " * x2)\n";

        cout << "\nSimulacion de nuevos experimentos:\n";
        cout << "----------------------------------------\n";

        double x1_test[5] = {50, 60, 55, 70, 65};
        double x2_test[5] = {30, 31, 32, 33, 31};

        for(int i = 0; i < 5; i++) {
            double y_pred = predict(x1_test[i], x2_test[i]);

            cout << "Experimento " << i+1 << ":\n";
            cout << "  x1 = " << x1_test[i]
                 << ", x2 = " << x2_test[i] << endl;
            cout << "  Yield predicho = " << y_pred << endl;
            cout << "----------------------------------------\n";
        }
    }
};

int main() {
    Dataset data;
    LinearRegression model;

    model.fit(data);
    model.printResults();

    return 0;
}
