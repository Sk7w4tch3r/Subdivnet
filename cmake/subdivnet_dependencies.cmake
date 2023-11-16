function (subdivnet_find_dependencies_thirdparty)
endfunction()

function (subdivnet_find_dependencies_internal)
endfunction()

function (subdivnet_dependency_targets)
    set(SUBDIVNET_DEPENDENCY_TARGETS
        PARENT_SCOPE
    )
    list(REMOVE_DUPLICATES SUBDIVNET_DEPENDENCY_TARGETS)
endfunction()
