#include <half.hpp>

#include <iostream>
using half_float::half;

int main()
{
	half a(3.2), b(5);
	half c = a * b;
	c += 3;
	if(c > a)
    		std::cout << c << std::endl;
}

