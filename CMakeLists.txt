cmake_minimum_required(VERSION 3.13.0)

project(em_wave_pml LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置默认构建类型为 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release."
        FORCE)
endif()

# 配置 OpenBLAS
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# 确保使用串行版本的 deal.II
set(DEAL_II_WITH_MPI OFF CACHE BOOL "" FORCE)

# 禁用 MKL
set(DEAL_II_WITH_MKL OFF CACHE BOOL "" FORCE)

# 设置链接标志
set(DEAL_II_LINKER_FLAGS "${DEAL_II_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
set(DEAL_II_LIBRARIES "${DEAL_II_LIBRARIES};${BLAS_LIBRARIES};${LAPACK_LIBRARIES}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 查找 deal.II 包
find_package(deal.II 9.0
    HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
    REQUIRED
)

# 检查是否找到了deal.II
IF(NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
        "*** Could not locate deal.II. ***\n"
        "*** You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
        "*** or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

# 初始化 deal.II 的缓存变量
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# 添加可执行文件
add_executable(${PROJECT_NAME} em_wave.cpp)

# 根据 CMAKE_BUILD_TYPE 设置相应的 DEAL_II_SETUP_TARGET 模式
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    DEAL_II_SETUP_TARGET(${PROJECT_NAME} DEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE -g -O0)
else()
    DEAL_II_SETUP_TARGET(${PROJECT_NAME} RELEASE)
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
endif()