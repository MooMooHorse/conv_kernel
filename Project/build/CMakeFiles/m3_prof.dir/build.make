# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /ece408/project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /build

# Include any dependencies generated for this target.
include CMakeFiles/m3_prof.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/m3_prof.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/m3_prof.dir/flags.make

CMakeFiles/m3_prof.dir/m3_prof.cc.o: CMakeFiles/m3_prof.dir/flags.make
CMakeFiles/m3_prof.dir/m3_prof.cc.o: /ece408/project/m3_prof.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/m3_prof.dir/m3_prof.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/m3_prof.dir/m3_prof.cc.o -c /ece408/project/m3_prof.cc

CMakeFiles/m3_prof.dir/m3_prof.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/m3_prof.dir/m3_prof.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ece408/project/m3_prof.cc > CMakeFiles/m3_prof.dir/m3_prof.cc.i

CMakeFiles/m3_prof.dir/m3_prof.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/m3_prof.dir/m3_prof.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ece408/project/m3_prof.cc -o CMakeFiles/m3_prof.dir/m3_prof.cc.s

CMakeFiles/m3_prof.dir/m3_prof.cc.o.requires:

.PHONY : CMakeFiles/m3_prof.dir/m3_prof.cc.o.requires

CMakeFiles/m3_prof.dir/m3_prof.cc.o.provides: CMakeFiles/m3_prof.dir/m3_prof.cc.o.requires
	$(MAKE) -f CMakeFiles/m3_prof.dir/build.make CMakeFiles/m3_prof.dir/m3_prof.cc.o.provides.build
.PHONY : CMakeFiles/m3_prof.dir/m3_prof.cc.o.provides

CMakeFiles/m3_prof.dir/m3_prof.cc.o.provides.build: CMakeFiles/m3_prof.dir/m3_prof.cc.o


# Object files for target m3_prof
m3_prof_OBJECTS = \
"CMakeFiles/m3_prof.dir/m3_prof.cc.o"

# External object files for target m3_prof
m3_prof_EXTERNAL_OBJECTS =

m3_prof: CMakeFiles/m3_prof.dir/m3_prof.cc.o
m3_prof: CMakeFiles/m3_prof.dir/build.make
m3_prof: /usr/local/cuda/lib64/libcudart_static.a
m3_prof: /usr/lib/x86_64-linux-gnu/librt.so
m3_prof: libece408net.a
m3_prof: src/libMiniDNNLib.a
m3_prof: src/libGpuConv.a
m3_prof: /usr/local/cuda/lib64/libcudart_static.a
m3_prof: /usr/lib/x86_64-linux-gnu/librt.so
m3_prof: CMakeFiles/m3_prof.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable m3_prof"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/m3_prof.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/m3_prof.dir/build: m3_prof

.PHONY : CMakeFiles/m3_prof.dir/build

CMakeFiles/m3_prof.dir/requires: CMakeFiles/m3_prof.dir/m3_prof.cc.o.requires

.PHONY : CMakeFiles/m3_prof.dir/requires

CMakeFiles/m3_prof.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/m3_prof.dir/cmake_clean.cmake
.PHONY : CMakeFiles/m3_prof.dir/clean

CMakeFiles/m3_prof.dir/depend:
	cd /build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /ece408/project /ece408/project /build /build /build/CMakeFiles/m3_prof.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/m3_prof.dir/depend

