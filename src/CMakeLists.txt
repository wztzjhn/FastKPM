set(KPM_SRC
  spmat.cpp
  fastkpm.cpp
  engine.cpp
  engine_cpu.cpp
  engine_cusparse.cpp
  engine_mpi.cpp
  timer.cpp
)

set(KPM_INCLUDE
  fastkpm.h
)

if(WITH_CUDA)
    add_library(fastkpm SHARED outer_product.cu ${KPM_SRC})

    target_compile_options(fastkpm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)

    # Disable host flags propagation by explicitly setting CUDA flags
    # PASCAL (60,61); VOLTA (70); TURING (75); AMPERE (80,86)
    set_target_properties(fastkpm PROPERTIES
        CUDA_ARCHITECTURES "60;61;70;75;80;86"
        PUBLIC_HEADER "${KPM_INCLUDE}"
        CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(fastkpm PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

    # Link system libraries as shared libraries
    target_link_options(fastkpm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xlinker=-lpthread -Xlinker=-lrt -Xlinker=-ldl>)

    # Link remaining libraries
    target_link_libraries(fastkpm PRIVATE ${LIBS})
else()
    add_library(fastkpm SHARED ${KPM_SRC})
    set_target_properties(fastkpm PROPERTIES PUBLIC_HEADER "${KPM_INCLUDE}")
    target_link_libraries(fastkpm ${LIBS})
endif()

# Add the fastkpm library and headers to the "export-set"
install(TARGETS fastkpm
  EXPORT fastkpmTargets
  LIBRARY       DESTINATION "${INSTALL_LIB_DIR}"     COMPONENT shlib
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT dev
  INCLUDES      DESTINATION include
)

