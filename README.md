# DPT
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
clang -o printdevices printdevices.c -framework OpenCL

clang -o matmul matmul.c -framework OpenCL
clang -o simple simple.c -framework OpenCL
clang -o square_direct square_direct.c -framework OpenCL
clang -o square square.c -framework OpenCL
clang -o timer timer.c -framework OpenCL
clang -o transpose transpose.c -framework OpenCL

