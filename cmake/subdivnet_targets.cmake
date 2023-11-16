include(subdivnet_dependencies)

function (subdivnet_declare_lib)
    subdivnet_dependency_targets()
    # add generated files to library
    set(BASE_DIR "${PROJECT_SOURCE_DIR}/")
    add_library(
        subdivnet_lib OBJECT
        "${BASE_DIR}/source/conv.cpp"
        "${BASE_DIR}/source/conv.h"
        "${BASE_DIR}/source/subdivnet/dataset.cpp"
        "${BASE_DIR}/source/subdivnet/dataset.h"
        "${BASE_DIR}/source/subdivnet/deeplab.cpp"
        "${BASE_DIR}/source/subdivnet/deeplab.h"
        "${BASE_DIR}/source/subdivnet/mesh_ops.cpp"
        "${BASE_DIR}/source/subdivnet/mesh_ops.h"
        "${BASE_DIR}/source/subdivnet/mesh_tensor.cpp"
        "${BASE_DIR}/source/subdivnet/mesh_tensor.h"
        "${BASE_DIR}/source/subdivnet/network.cpp"
        "${BASE_DIR}/source/subdivnet/network.h"
        "${BASE_DIR}/source/subdivnet/utils.h"
        "${BASE_DIR}/source/maps/geometry.cpp"
        "${BASE_DIR}/source/maps/geometry.h"
        "${BASE_DIR}/source/maps/maps.cpp"
        "${BASE_DIR}/source/maps/maps.h"
        "${BASE_DIR}/source/maps/utils.h"
    )

    # includes
    target_include_directories(
        subdivnet_lib SYSTEM ${warning_guard}
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
        ${CMAKE_CURRENT_BINARY_DIR}
    )

    target_compile_features(subdivnet_lib PUBLIC cxx_std_20)
    target_link_libraries(subdivnet_lib PRIVATE ${SUBDIVNET_DEPENDENCY_TARGETS})

endfunction()

function (subdivnet_declare_exe BASE_DIR)
    subdivnet_dependency_targets()

    add_executable(subdivnet_exe
        "${BASE_DIR}source/main.cpp"
        "${BASE_DIR}source/vis/model.cpp"
        "${BASE_DIR}source/vis/model.h"
    )
    add_executable(subdivnet::exe ALIAS subdivnet_exe)
    set_property(TARGET subdivnet_exe PROPERTY OUTPUT_NAME subdivnet)
    target_include_directories(
        subdivnet_exe SYSTEM ${warning_guard}
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
        ${CMAKE_CURRENT_BINARY_DIR}
    )
    target_compile_features(subdivnet_exe PRIVATE cxx_std_20)
    target_link_libraries(subdivnet_exe PRIVATE subdivnet_lib ${SUBDIVNET_DEPENDENCY_TARGETS})
endfunction()
