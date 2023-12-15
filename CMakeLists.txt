cmake_minimum_required(VERSION 3.22)

# to be changed later
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CUDACXX "/usr/local/cuda/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/")

project(starter)
# enable_language(CUDA)

include(cmake/prelude.cmake)

project(
    subdivnet
    VERSION 0.1.0
    LANGUAGES CXX C
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "/home/hesam/miniconda3/lib/python3.11/site-packages/torch/share/cmake") # to be changed later

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)
include(subdivnet_dependencies)
include(subdivnet_targets)

# ---- Dependencies ----
subdivnet_find_dependencies_thirdparty()
subdivnet_find_dependencies_internal()
subdivnet_dependency_targets()

# ---- Declare library ----
subdivnet_declare_lib()


# find cgal and its components
find_package(givdeapp REQUIRED)
find_package(givdelib REQUIRED)
find_package(givdetypes REQUIRED)
find_package(givdefp REQUIRED)
# find_package(givderender REQUIRED)
# find_package(givdegeometry REQUIRED)
find_package(CGAL REQUIRED COMPONENTS Core)
find_package(nlohmann_json REQUIRED)
find_package(nlohmann_json_schema_validator REQUIRED)
find_package(Torch REQUIRED)
find_package(libigl REQUIRED)
find_package(libpqxx REQUIRED)
find_package(cereal REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


list(APPEND dependencies givdeapp::givdeapp)
list(APPEND dependencies givdelib::givdelib)
list(APPEND dependencies givdetypes::givdetypes)
list(APPEND dependencies givdefp::givdefp)
# list(APPEND dependencies givderender::givderender)
# list(APPEND dependencies givdegeometry::givdegeometry)
list(APPEND dependencies CGAL::CGAL)
list(APPEND dependencies nlohmann_json::nlohmann_json)
list(APPEND dependencies nlohmann_json_schema_validator)
# use libigl
list(APPEND dependencies libigl::libigl)
list(APPEND dependencies libpqxx::pqxx)
list(APPEND dependencies cereal::cereal)
list(APPEND dependencies "${TORCH_LIBRARIES}")



# includes
target_include_directories(
    subdivnet_lib SYSTEM ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
    submodules/lager
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_features(subdivnet_lib PUBLIC cxx_std_20)
target_link_libraries(subdivnet_lib PRIVATE ${dependencies})

# ---- Declare executable ----

add_executable(subdivnet_exe source/preprocess.cpp)
add_executable(train_subdivnet_exe source/train_cls.cpp)
add_executable(subdivnet::exe ALIAS subdivnet_exe)
set_property(TARGET subdivnet_exe PROPERTY OUTPUT_NAME subdivnet)
set_property(TARGET train_subdivnet_exe PROPERTY OUTPUT_NAME train_subdivnet)
target_compile_features(subdivnet_exe PRIVATE cxx_std_20)
target_compile_features(train_subdivnet_exe PRIVATE cxx_std_20)
target_link_libraries(subdivnet_exe PRIVATE subdivnet_lib ${dependencies})
target_link_libraries(train_subdivnet_exe PRIVATE subdivnet_lib ${dependencies})

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT subdivnet_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of subdivnet"
  )
endif()

include(cmake/dev-mode.cmake)
