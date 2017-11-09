#include <thrust/version.h>
#include <iostream>

/* Version check for thrust
   If not found, try nvcc version.cu -o version -I /home/you/libraries/
   when libraries is where you store you thrust downloaded files
*/

int main(void)
{
      int major = THRUST_MAJOR_VERSION;
        int minor = THRUST_MINOR_VERSION;

          std::cout << "Thrust v" << major << "." << minor << std::endl;

            return 0;
}
