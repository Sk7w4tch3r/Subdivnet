install(
    TARGETS subdivnet_exe
    RUNTIME COMPONENT subdivnet_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
