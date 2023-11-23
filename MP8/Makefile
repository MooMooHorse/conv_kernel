# Name of this mp
MP_NAME = mp8

# define LIBWB_PATH
LIBWB_PATH = /home/hao/23fall/ece-408/libwb
# SO file path
LIBWB_SO_PATH = $(LIBWB_PATH)/lib/libwb.so

MP_OBJ = $(MP_NAME).o

$(MP_NAME): $(MP_OBJ)
	nvcc -std=c++11 -o $(MP_NAME) $(MP_OBJ) $(LIBWB_SO_PATH)
$(MP_OBJ): template.cu
	nvcc -std=c++11 -I$(LIBWB_PATH) -c template.cu -o $(MP_OBJ)
# run the program
PHONY: run
run:
	bash run_datasets

# clean
PHONY: clean
clean:
	rm -f $(MP_NAME) *.o