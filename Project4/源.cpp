#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>

using namespace sycl;

#define n 512

int main() {
    // 创建用于在 GPU 上执行的队列
    queue q(gpu_selector{});

    // 定义矩阵 A 的缓冲区
    int A[n][n];

    // 初始化矩阵 A
    for (int i = 0; i < n; i++) {
        A[i][i] = 1.0;
        for (int j = 0; j < n; j++) {
            if (j > i)
                A[i][j] = rand() % 10;
            else if (j < i)
                A[i][j] = 0;
        }
    }

    // 创建用于在设备上存储矩阵 A 的缓冲区
    buffer<int, 2> bufA(reinterpret_cast<int*>(A), range<2>(n, n));

    // 启动计时器
    auto start = std::chrono::steady_clock::now();

    // 提交一个命令组，在设备上执行
    q.submit([&](handler& h) {
        // 访问缓冲区
        auto accA = bufA.get_access<access::mode::read_write>(h);

        // 在设备上执行矩阵操作
        h.parallel_for(range<1>(n), [=](id<1> idx) {
            int k = idx[0];

            // 对第 k 行进行归一化
            for (int j = k + 1; j < n; j++) {
                accA[k][j] = accA[k][j] / accA[k][k];
            }
            accA[k][k] = 1.0;

            // 执行高斯消元
            for (int i = k + 1; i < n; i++) {
                for (int j = k + 1; j < n; j++) {
                    accA[i][j] = accA[i][j] - accA[k][j] * accA[i][k];
                }
                accA[i][k] = 0;
            }
            });
        });

    // 将修改后的矩阵 A 读回主机
    q.wait();
    bufA.get_access<access::mode::read>();

    // 停止计时器并计算经过的时间
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // 打印结果矩阵
    std::cout << "结果矩阵:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A[i][j] << " ";
        }
        std::cout << "\n";
    }

    // 打印经过的时间
    std::cout << "经过时间: " << duration << " 微秒\n";

    return 0;
}




