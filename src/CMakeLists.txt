set(KPM_SRC
  spmat.cpp
  fastkpm.cpp
  engine.cpp
  engine_cpu.cpp
  engine_cusparse.cpp
  timer.cpp
)

set(KPM_INCLUDE
  fastkpm.h
)

if (CUDA_FOUND)
  cuda_add_library(fastkpm SHARED outer_product.cu ${KPM_SRC})
else()
  add_library(fastkpm SHARED ${KPM_SRC})
endif()
target_link_libraries(fastkpm ${LIBS})
set_target_properties(fastkpm PROPERTIES PUBLIC_HEADER "${KPM_INCLUDE}")

# Add the fastkpm library and headers to the "export-set"
install(TARGETS fastkpm
  EXPORT FastKPMTargets
  LIBRARY       DESTINATION "${INSTALL_LIB_DIR}"     COMPONENT shlib
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT dev)
