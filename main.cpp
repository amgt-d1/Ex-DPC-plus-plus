#include "exdpc.hpp"
#include <time.h>


int main()
{
    Eigen::setNbThreads(1);
    //std::cout << " Number of OpenMP threads: " << Eigen::nbThreads() << "\n";

    EX_DPC exdpc;
    exdpc.input_parameter();
    exdpc.input_data();

    time_t t = time(NULL);
	printf(" %s\n\n", ctime(&t));

    // clustering
    exdpc.run();

    // result output
    //exdpc.output_label();
    //exdpc.output_decision_graph();
    exdpc.output_result();

    return 0;
}
