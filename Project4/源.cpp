#include <iostream>
#include<ctime>
#include<Windows.h>
#include <immintrin.h>
#include <intrin.h>
#include<iomanip>
using namespace std;
double time1 = 0, time2 = 0, time3 = 0, time4 = 0;
void test() {
    int n = 512;//n为数据规模，取64、128、256、512、1024五个值
    float** m = new float* [n];//串行矩阵
    float** A = new float* [n];//二重循环SSE矩阵
    float** A1 = new float* [n];//三重循环SSE矩阵
    float** A2 = new float* [n];//SSE优化矩阵
    srand(time(0));
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        m[i] = new float[n];
        A1[i] = new float[n];
        A2[i] = new float[n];
    }
    for (int i = 0; i < n; i++) {
        A[i][i] = 1.0;//对角线为1.0
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 10;
        }

    }
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] += A[k][j];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i][j] = A[i][j];//两个完全相同的矩阵，一个用于串行算法，一个用于二重循环的SSE向量化
            A1[i][j] = A[i][j];//两个完全相同的矩阵，一个用于二重循环的SSE向量化，一个用于三重循环的SSE向量化
            A2[i][j] = A[i][j];//A2为完全SSE向量化
        }
    }
    //测试用例的生成


    long long head1, tail1, freq1; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head1);

    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            }
            m[i][k] = 0;
        }
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
    cout << "数据规模为n=" << n << endl;
    time1 += (tail1 - head1) * 1000.0 / freq1;
    cout << "串行算法测试时间为 " << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;


    /////////////////////////////////////////////////////////////////////二重循环SSE向量化//////////////////////////////////////////////////////////////////////////////////////////
    long long head2, tail2, freq2; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head2);

    for (int k = 0; k < n; k++) {
        float vt = A1[k][k];//vt ← dupTo4Float(A[k,k]);
        __m128 va;//float型寄存器
        for (int j = k + 1; j + 4 < n; j += 4) {
            va = _mm_load_ps(&A1[k][j]);//va ← load4FloatFrom(&A[k,j]) ; 
            va = _mm_div_ps(va, _mm_set1_ps(vt)); //va ← va/vt ; // 这里是向量对位相除
            _mm_store_ps(&A1[k][j], va);//store4FloatTo(&A[k, j], va);
        }
        //4路向量化
        A1[k][k] = 1.0;//A[k,k] ← 1.0;

        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A1[i][j] = A1[i][j] - A1[k][j] * A1[i][k];
            }
            A1[i][k] = 0;
        }
    }


    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
    cout << "SSE二重循环向量化运行时间为 " << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;
    time2 += (tail2 - head2) * 1000.0 / freq2;

    /////////////////////////////////////////////////////////////////////三重循环SSE向量化//////////////////////////////////////////////////////////////////////////////////////////

    long long head3, tail3, freq3; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq3);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head3);

    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            A2[k][j] = A2[k][j] / A2[k][k];
        }
        A2[k][k] = 1.0;//A[k,k] ← 1.0;

        __m128 vaik, vakj, vaij, vx;//float型寄存器
        for (int i = k + 1; i < n; i++) {
            vaik = _mm_set1_ps(A2[i][k]);//vaik ← dupToVector4(A[i,k]);
            for (int j = k + 1; j + 4 < n; j += 4) {
                vakj = _mm_load_ps(&A2[k][j]);//vakj ← load4FloatFrom(&A[k,j]);
                vaij = _mm_load_ps(&A2[i][j]);//vaij ← load4FloatFrom(&A[i,j]);
                vx = _mm_mul_ps(vakj, vaik);//vx ← vakj*vaik;
                vaij = _mm_sub_ps(vaij, vx);//vaij ← vaij-vx;
                _mm_store_ps(&A2[i][j], vaij);//store4FloatTo(&A[i,j],vaij);
            }
            A2[i][k] = 0;//A[i,k] ← 0;
        }
    }


    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail3);
    cout << "SSE三重循环向量化运行时间为 " << (tail3 - head3) * 1000.0 / freq3 << "ms" << endl;
    time3 += (tail3 - head3) * 1000.0 / freq3;

    /////////////////////////////////////////////////////////////////////完全SE向量化//////////////////////////////////////////////////////////////////////////////////////////

    long long head, tail, freq; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    for (int k = 0; k < n; k++) {
        float vt = A[k][k];//vt ← dupTo4Float(A[k,k]);
        __m128 va;//float型寄存器
        for (int j = k + 1; j + 4 < n; j += 4) {
            va = _mm_load_ps(&A[k][j]);//va ← load4FloatFrom(&A[k,j]) ; 
            va = _mm_div_ps(va, _mm_set1_ps(vt)); //va ← va/vt ; // 这里是向量对位相除
            _mm_store_ps(&A[k][j], va);//store4FloatTo(&A[k, j], va);
        }
        //4路向量化
        A[k][k] = 1.0;//A[k,k] ← 1.0;

        __m128 vaik, vakj, vaij, vx;//float型寄存器
        for (int i = k + 1; i < n; i++) {
            vaik = _mm_set1_ps(A[i][k]);//vaik ← dupToVector4(A[i,k]);
            for (int j = k + 1; j + 4 < n; j += 4) {
                vakj = _mm_load_ps(&A[k][j]);//vakj ← load4FloatFrom(&A[k,j]);
                vaij = _mm_load_ps(&A[i][j]);//vaij ← load4FloatFrom(&A[i,j]);
                vx = _mm_mul_ps(vakj, vaik);//vx ← vakj*vaik;
                vaij = _mm_sub_ps(vaij, vx);//vaij ← vaij-vx;
                _mm_store_ps(&A[i][j], vaij);//store4FloatTo(&A[i,j],vaij);
            }
            A[i][k] = 0;//A[i,k] ← 0;
        }
    }


    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SSE完全向量化运行时间为 " << (tail - head) * 1000.0 / freq << "ms" << endl;
    time4 += (tail - head) * 1000.0 / freq;

    cout << endl;
    cout << endl;

}
//以上为SSE向量化
void test1() {
    int n = 512;//n为数据规模，取64、128、256、512、1024五个值
    float** m = new float* [n];//串行矩阵
    float** A = new float* [n];//二重循环SSE矩阵
    float** A1 = new float* [n];//三重循环SSE矩阵
    float** A2 = new float* [n];//SSE优化矩阵
    srand(time(0));
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        m[i] = new float[n];
        A1[i] = new float[n];
        A2[i] = new float[n];
    }
    for (int i = 0; i < n; i++) {
        A[i][i] = 1.0;//对角线为1.0
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 10;
        }

    }
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] += A[k][j];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i][j] = A[i][j];//两个完全相同的矩阵，一个用于串行算法，一个用于二重循环的AVX向量化
            A1[i][j] = A[i][j];//两个完全相同的矩阵，一个用于二重循环的SSE向量化，一个用于三重循环的AVX向量化
            A2[i][j] = A[i][j];//A2为完全AVX向量化
        }
    }
    //测试用例的生成

    long long head1, tail1, freq1; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head1);

    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            }
            m[i][k] = 0;
        }
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
    cout << "数据规模为n=" << n << endl;
    time1 += (tail1 - head1) * 1000.0 / freq1;
    cout << "串行算法测试时间为 " << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;


    /////////////////////////////////////////////////////////////////////二重循环SSE向量化//////////////////////////////////////////////////////////////////////////////////////////

    long long head2, tail2, freq2; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head2);

    for (int k = 0; k < n; k++) {
        float vt = A1[k][k];
        __m256 va;
        for (int j = k + 1; j + 8 <= n; j += 8) {
            va = _mm256_load_ps(&A1[k][j]);
            va = _mm256_div_ps(va, _mm256_set1_ps(vt));
            _mm256_store_ps(&A1[k][j], va);
        }
        A1[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A1[i][j] = A1[i][j] - A1[k][j] * A1[i][k];
            }
            A1[i][k] = 0;
        }
    }

    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
    cout << "AVX二重循环向量化运行时间为 " << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;
    time2 += (tail2 - head2) * 1000.0 / freq2;

    /////////////////////////////////////////////////////////////////////三重循环SSE向量化//////////////////////////////////////////////////////////////////////////////////////////

    long long head3, tail3, freq3; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq3);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head3);

    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            A2[k][j] = A2[k][j] / A2[k][k];
        }
        A2[k][k] = 1.0;

        __m256 vaik, vakj, vaij, vx;
        for (int i = k + 1; i < n; i++) {
            vaik = _mm256_set1_ps(A2[i][k]);
            for (int j = k + 1; j + 8 <= n; j += 8) {
                vakj = _mm256_load_ps(&A2[k][j]);
                vaij = _mm256_load_ps(&A2[i][j]);
                vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_store_ps(&A2[i][j], vaij);
            }
            A2[i][k] = 0;
        }
    }


    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail3);
    cout << "AVX三重循环向量化运行时间为 " << (tail3 - head3) * 1000.0 / freq3 << "ms" << endl;
    time3 += (tail3 - head3) * 1000.0 / freq3;

    /////////////////////////////////////////////////////////////////////完全SE向量化//////////////////////////////////////////////////////////////////////////////////////////

    long long head, tail, freq; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    for (int k = 0; k < n; k++) {
        float vt = A[k][k];
        __m256 va;
        for (int j = k + 1; j + 8 <= n; j += 8) {
            va = _mm256_load_ps(&A[k][j]);
            va = _mm256_div_ps(va, _mm256_set1_ps(vt));
            _mm256_store_ps(&A[k][j], va);
        }
        A[k][k] = 1.0;

        __m256 vaik, vakj, vaij, vx;
        for (int i = k + 1; i < n; i++) {
            vaik = _mm256_set1_ps(A[i][k]);
            for (int j = k + 1; j + 8 <= n; j += 8) {
                vakj = _mm256_load_ps(&A[k][j]);
                vaij = _mm256_load_ps(&A[i][j]);
                vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_store_ps(&A[i][j], vaij);
            }
            A[i][k] = 0;
        }
    }


    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "AVX完全向量化运行时间为 " << (tail - head) * 1000.0 / freq << "ms" << endl;
    time4 += (tail - head) * 1000.0 / freq;

    cout << endl;
    cout << endl;
    cout << endl;
}
int main() {

    for (int i = 1; i <= 20; i++) {
        test1();
    }

    //cout << "串行算法平均运行时间为" << time1/20 << "ms" << endl;
    //cout << "SSE二重循环向量化平均运行时间为 " << time2/20 << "ms" << endl;
    //cout << "SSE三重循环向量化平均运行时间为 " << time3/20 << "ms" << endl;
    //cout << "SSE完全向量化平均运行时间为 " << time4/20<<"ms" << endl;



    cout << "串行算法平均运行时间为" << time1 / 20 << "ms" << endl;
    cout << "AVX二重循环向量化平均运行时间为 " << time2 / 20 << "ms" << endl;
    cout << "AVX三重循环向量化平均运行时间为 " << time3 / 20 << "ms" << endl;
    cout << "AVX完全向量化平均运行时间为 " << time4 / 20 << "ms" << endl;
    return 0;
}
