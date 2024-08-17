#include <iostream>
#include "../include/minigpu.h"

void testCreateContext()
{
    mgpuInitializeContext();
}

// Add more test functions for other functions in minigpu.cpp

int main()
{
    testCreateContext();
    // Call other test functions

    return 0;
}
