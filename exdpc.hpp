#pragma once
#define EIGEN_DONT_PARALLELIZE

# include "vptree.hpp"
# include "utils.h"
# include <iostream>
# include <exception>

#include <random>
#include <chrono>
#include <unistd.h>
#include <unordered_map>


// compute memory usage
double process_mem_usage()
{
    double resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    resident_set = rss * page_size_kb;

	return resident_set / 1000;
}


class EX_DPC
{
    // system paramter
    unsigned int dataset_id = 0;
    unsigned int thread_num = 1;
    float sampling_rate = 1.0;
    unsigned int dimensionality = 4;
    unsigned int k = 50;

    // DPC parameter
    float cutoff = 0;
    float local_density_min = 0;
    float delta_min = 0;

    // dataset + index
    std::vector<pointType> point_set;
    std::vector<std::vector<float>> pset;
    vp_tree vpt;
    CoverTree* ct = NULL;

    // local density
    std::vector<std::pair<float, unsigned int>> local_density_set;

    // dependent distance
    std::vector<float> dependent_distance_set;
    std::unordered_map<unsigned int, std::vector<unsigned int>> dependency_reverse;
    std::vector<int> dependent_point;

    // label
    std::vector<int> labels;

    // kNN matrix
    std::vector<std::vector<unsigned int>> knn_set;

    // result
    float time_offline = 0;
    float time_local_density_comp = 0;
    float time_dependency_comp = 0;
    double memory_usage = 0;

    // distance computation
    double compute_distance(const pointType& l, const pointType& r)
    {
        float distance = 0;
        for (unsigned int i = 0; i < dimensionality; ++i) distance += (l.first[i] - r.first[i]) * (l.first[i] - r.first[i]);
        return sqrt(distance);
    }

    // build knn-index
    void build_knn_matrix()
    {
        _s = std::chrono::system_clock::now();

        const unsigned int size = pset.size();
        knn_set.resize(size);
        for (unsigned int i = 0; i < pset.size(); ++i) knn_set[i].resize(k);

        #pragma omp parallel num_threads(44)
        {
            #pragma omp for schedule(dynamic)
            for (unsigned int i = 0; i < size; ++i)
            {
                vpt.knn_search(pset[i], k, knn_set[i]);
            }
            //#pragma omp for schedule(dynamic)
            //for (unsigned int i = 0; i < size; ++i) {
            //    std::vector<std::pair<CoverTree::Node*, double>> knn = ct->kNearestNeighbours(point_set[i], k);
            //    for (unsigned int j = 0; j < k; ++j) knn_set[i][j] = knn[j].first->_p.second;
            //}
        }

        _e = std::chrono::system_clock::now();
		time_offline = std::chrono::duration_cast<std::chrono::microseconds>(_e - _s).count();
        time_offline /= 1000;
		std::cout << " offline time: " << time_offline << "[msec]\n";
    }

    // local density computation
    void compute_local_density()
    {
        _s = std::chrono::system_clock::now();

        const unsigned int size = point_set.size();
        local_density_set.resize(size);

        #pragma omp parallel num_threads(thread_num)
        {
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < size; ++i)
            {
                std::mt19937 mt(i);
                std::uniform_real_distribution<> rnd(0, 0.9999);
                local_density_set[i] = {vpt.range_count(pset[i], i, cutoff) + rnd(mt), i};
            }
        }

        _e = std::chrono::system_clock::now();
		time_local_density_comp = std::chrono::duration_cast<std::chrono::microseconds>(_e - _s).count();
        time_local_density_comp /= 1000;
		std::cout << " local density comp. time: " << time_local_density_comp << "[msec]\n";

        float avg = 0;
        for (unsigned int i = 0; i < size; ++i) avg += local_density_set[i].first;
        avg /= size;
        std::cout << " average local density: " << avg << "\n\n";
    }

    // dependency computation
    void compute_dependency()
    {
        _s = std::chrono::system_clock::now();

        const unsigned int size = point_set.size();

        // sort by local density
        std::vector<std::pair<float, unsigned int>> local_density_set_cp = local_density_set;
        std::sort(local_density_set_cp.begin(), local_density_set_cp.end(), std::greater<std::pair<float, unsigned int>>());

        // resize dependency
        dependent_distance_set.resize(size);
        dependent_point.resize(size);

        const unsigned int size_div = ceil(sqrt((size/log2(size)) * pow(1.0,-12)));
        //std::cout << " subset size: " << size_div << "\n";
        std::vector<std::vector<pointType>> point_set_div;
        std::vector<std::vector<unsigned int>> idx_set_div;
        const unsigned int size_subset = 1 + (size / size_div);
        std::vector<std::pair<float, float>> local_density_max_min(size_div);

        std::vector<pointType> temp;
        std::vector<unsigned int> temp_idx;
        unsigned int cnt = 0;
        local_density_max_min[cnt].first = local_density_set_cp[0].first;
        unsigned int count = 0;

        for (unsigned int i = 0; i < size; ++i)
        {
            // count #noise
            if (local_density_set_cp[i].first < local_density_min) ++count;

            // init dependency
            dependent_distance_set[i] = FLT_MAX;
            dependent_point[i] = -1;

            // update max
            if (temp.size() == 0) local_density_max_min[cnt].first = local_density_set_cp[i].first;

            // get idx
            const unsigned int idx = local_density_set_cp[i].second;

            // insert
            temp.push_back(point_set[idx]);
            temp_idx.push_back(i);

            // insert into collection
            if (temp.size() == size_subset || i == size - 1)
            {
                point_set_div.push_back(temp);
                temp.clear();
                idx_set_div.push_back(temp_idx);
                temp_idx.clear();

                // update min
                local_density_max_min[cnt].second = local_density_set_cp[i].first;
                ++cnt;
            }
        }

        // build cover-trees
        std::vector<CoverTree*> cforest(size_div);
        #pragma omp parallel num_threads(thread_num)
        {
            #pragma omp for schedule(static)
            for (unsigned int i = 0; i < size_div; ++i) cforest[i] = CoverTree::from_points(point_set_div[i], -1, false);
        }

        //_e = std::chrono::system_clock::now();
		//time_dependency_comp = std::chrono::duration_cast<std::chrono::microseconds>(_e - _s).count();
        //time_dependency_comp /= 1000;
		//std::cout << " build cover-tree time: " << time_dependency_comp << "[msec]\n\n";

        // compute dependent distance
        #pragma omp parallel num_threads(thread_num)
        {
            #pragma omp for schedule(dynamic)
            for (unsigned int i = 1; i < size - count; ++i)
            {
                // get idx
                const unsigned int idx = local_density_set_cp[i].second;

                // get local density
                const float local_density = local_density_set_cp[i].first;  

                float dependent_distance = FLT_MAX;
                int dependent_idx = -1;
                bool flag = 1;
                
                // kNN check
                for (unsigned int j = 0; j < k; ++j)
                {
                    const unsigned int _idx = knn_set[idx][j];
                    if (local_density < local_density_set[_idx].first)
                    {
                        flag = 0;
                        dependent_distance = compute_distance(point_set[idx], point_set[_idx]);
                        dependent_idx = _idx;
                        break;
                    }
                }
                
                if (flag)
                {
                    for (unsigned int j = 0; j < size_div; ++j)
                    {
                        bool type = 0;
                        if (local_density_max_min[j].first <= local_density)
                        {
                            break;
                        }
                        else
                        {
                            if (local_density_max_min[j].second > local_density) type = 1;
                        }

                        if (type)
                        {
                            // NNS on cover-tree
                            std::pair<CoverTree::Node*, double> ct_nn = cforest[j]->NearestNeighbour(point_set[idx]);
                            if (ct_nn.second < dependent_distance)
                            {
                                dependent_idx = ct_nn.first->_p.second;
                                dependent_distance = ct_nn.second;
                            }
                        }
                        else
                        {
                            // NNS by linear scan
                            const unsigned int s = point_set_div[j].size();
                            
                            for (unsigned int l = 0; l < s; ++l)
                            {
                                // get idx of local density
                                const unsigned int idx_ = idx_set_div[j][l];
                                if (local_density < local_density_set_cp[idx_].first)
                                {
                                    // distance computation
                                    float distance = compute_distance(point_set[idx], point_set[local_density_set_cp[idx_].second]);
                                    if (distance < dependent_distance)
                                    {
                                        dependent_distance = distance;
                                        dependent_idx = local_density_set_cp[idx_].second;
                                    }
                                }
                                else
                                {
                                    break;
                                }
                            }
                        }
                    }
                }

                dependent_distance_set[idx] = dependent_distance;
                dependent_point[idx] = dependent_idx;
                //if (i % 100000 == 0) std::cout << " " << i << "-th iteration done\n";
                //std::cout << i << "-th dependent distance: " << dependent_distance << "\t local density: " << local_density_set_cp[i].first << "\t idx: " << idx << "\t flag: " << flag << "\t last type: " << type_last << "\n";
            }
        }

        _e = std::chrono::system_clock::now();
		time_dependency_comp = std::chrono::duration_cast<std::chrono::microseconds>(_e - _s).count();
        time_dependency_comp /= 1000;
		std::cout << " dependency comp. time: " << time_dependency_comp << "[msec]\n\n";
    }

    // labeling
    void labeling()
    {
        // get size
        const unsigned int size = point_set.size();

        // resize label
        labels.resize(size);
        unsigned int cnt = 0;
        std::vector<unsigned int> centers;

        std::unordered_map<unsigned int, std::vector<unsigned int>> dependency_dic;
        for (unsigned int i = 0; i < size; ++i)
        {
            // init as noise
            labels[i] = -1;

            // get centers
            if (local_density_set[i].first >= local_density_min)
            {
                if (dependent_distance_set[i] >= delta_min)
                {
                    centers.push_back(i);

                    // update dependent point
                    dependent_point[i] = i;

                    // update label
                    labels[i] = cnt;
                    ++cnt;
                }
                else
                {
                    // make dic
                    if (local_density_set[i].first > local_density_min) dependency_dic[dependent_point[i]].push_back(i);
                }
            }
        }

        // init stack
        std::deque<unsigned int> stack;
        for (unsigned int i = 0; i < centers.size(); ++i) stack.push_back(centers[i]);

        // depth-first traversal
        while (stack.size() > 0)
        {
            // get index of top element
            const unsigned int idx = stack[0];

            // set label
            const int label = labels[idx];

            // delete top element
            stack.pop_front();

            if (dependency_dic.find(idx) != dependency_dic.end())
            {
                for (unsigned int i = 0; i < dependency_dic[idx].size(); ++i)
                {
                    // get idx
                    const unsigned int _idx = dependency_dic[idx][i];

                    if (labels[_idx] == -1)
                    {
                        // propagate label
                        labels[_idx] = label;

                        // update stack
                        stack.push_front(_idx);
                    }
                }
            }
        }
    }

public:
    
    // constructor
    EX_DPC() {}

    // parameter input
    void input_parameter()
    {
        std::ifstream ifs_cutoff("parameter/cutoff.txt");
        std::ifstream ifs_dataset_id("parameter/dataset_id.txt");
        std::ifstream ifs_thread_num("parameter/thread_num.txt");
        std::ifstream ifs_sampling_rate("parameter/sampling_rate.txt");
        std::ifstream ifs_k("parameter/k.txt");

        if (ifs_cutoff.fail())
        {
            std::cout << " cutoff.txt does not exist." << std::endl;
            std::exit(0);
        }
        else if (ifs_dataset_id.fail())
        {
            std::cout << " dataset_id.txt does not exist." << std::endl;
            std::exit(0);
        }
        else if (ifs_thread_num.fail())
        {
            std::cout << " thread_num.txt does not exist." << std::endl;
            std::exit(0);
        }
        else if (ifs_sampling_rate.fail())
        {
            std::cout << " sampling_rate.txt does not exist." << std::endl;
            std::exit(0);
        }
        else if (ifs_k.fail())
        {
            std::cout << " k.txt does not exist." << std::endl;
            std::exit(0);
        }

        while (!ifs_cutoff.eof()) { ifs_cutoff >> cutoff; }
        while (!ifs_dataset_id.eof()) { ifs_dataset_id >> dataset_id; }
        while (!ifs_thread_num.eof()) { ifs_thread_num >> thread_num; }
        while (!ifs_sampling_rate.eof()) { ifs_sampling_rate >> sampling_rate; }
        while (!ifs_k.eof()) { ifs_k >> k; }

        // determine delta_min & rho_min
        if (dataset_id == 0)
        {
            dimensionality = 3;
            delta_min = 25000;
            local_density_min = 200;
        }
        else if (dataset_id == 1)
        {
            dimensionality = 4;
            delta_min = 4000;
            local_density_min = 50;
        }
        else if (dataset_id == 2)
        {
            dimensionality = 4;
            delta_min = 10000;
            local_density_min = 100;
        }
        else if (dataset_id == 3)
        {
            dimensionality = 8;
            delta_min = 25000;
            local_density_min = 100;
        }
        else if (dataset_id == 4)
        {
            dimensionality = 2;
            delta_min = 2000;
            local_density_min = 100;
        }
        else if (dataset_id == 5)
        {
            dimensionality = 2;
            delta_min = 10000;
            local_density_min = 5;
	    }
        else if (dataset_id == 6)
        {
            dimensionality = 16;
            delta_min = 3000;
            local_density_min = 100;
	    }

        std::cout << " ---------\n";
        std::cout << " data id: " << dataset_id << "\n";
        std::cout << " dimensionality: " << dimensionality << "\n";
        std::cout << " sampling rate: " << sampling_rate << "\n";
        std::cout << " cutoff-distance: " << cutoff << "\n";
        std::cout << " #threads: " << thread_num << "\n";

        local_density_min *= sampling_rate;
    }

    // data input
    void input_data()
    {
        // id variable
        unsigned int id = 0;

        // position & id variables
        std::vector<float> d_max(dimensionality);
        std::vector<float> d_min(dimensionality);
        for (unsigned int i = 0; i < dimensionality; ++i) d_min[i] = FLT_MAX;

        // sample probability
        std::mt19937 mt(1);
        std::uniform_real_distribution<> rnd(0, 1.0);

        // dataset input
        std::string f_name = "dataset/";
        if (dataset_id == 0) f_name += "airline-3d.csv";
        if (dataset_id == 1) f_name += "household-4d.csv";
        if (dataset_id == 2) f_name += "pamap2-4d.csv";
        if (dataset_id == 3) f_name += "sensor-8d.csv";
        if (dataset_id == 4) f_name += "tdrive-2d.csv";
        if (dataset_id == 5) f_name += "syn-2d.csv";
        if (dataset_id == 6) f_name += "gas-16d.csv";

        // file input
        std::ifstream ifs_file(f_name);
        std::string full_data;

        // error check
        if (ifs_file.fail())
        {
            std::cout << " data file does not exist." << std::endl;
            std::exit(0);
        }

        // read data
        while (std::getline(ifs_file, full_data))
        {
            std::string meta_info;
            std::istringstream stream(full_data);
            std::string type = "";
            //pointType newPt = pointType(dimensionality);
            pointType newPt;
            newPt.first.resize(dimensionality);
            newPt.second = id;

            for (unsigned int i = 0; i < dimensionality; ++i)
            {
                std::getline(stream, meta_info, ',');
                std::istringstream stream_(meta_info);
                long double val = std::stold(meta_info);
                newPt.first[i] = val;

                if (val > d_max[i]) d_max[i] = val;
				if (val < d_min[i]) d_min[i] = val;
            }

            // insert into dataset
            if (rnd(mt) <= sampling_rate)
            {
                point_set.push_back(newPt);
                ++id;
            }
        }

        const unsigned int size = point_set.size();
        float coord_max = 100000;

        pset.resize(size);
        for (unsigned int i = 0; i < size; ++i) pset[i].resize(dimensionality);

        bool flag = 1;
        //if (dataset_id == 4 || dataset_id == 6) flag = 0;
        if (flag)
        {
            if (dataset_id == 0) coord_max *= 10;
            for (unsigned int i = 0; i < dimensionality; ++i) d_max[i] -= d_min[i];
            const unsigned int size = point_set.size();

            for (unsigned int i = 0; i < size; ++i)
            {
                pset[i].resize(dimensionality);
                for (unsigned int j = 0; j < dimensionality; ++j)
                {
                    point_set[i].first[j] -= d_min[j];
                    point_set[i].first[j] /= d_max[j];
                    point_set[i].first[j] *= coord_max;
                    pset[i][j] = point_set[i].first[j];
                }
            }
        }
        
        std::cout << " cardinality: " << size << "\n";
        std::cout << " ---------\n\n";
    }

    // run
    void run()
    {
        // offline
        memory_usage = process_mem_usage();
        vpt.input(pset, dimensionality);
        time_offline = vpt.build();
        std::cout << " vp-tree build time: " << time_offline << "[msec]\n";
        build_knn_matrix();
        memory_usage = process_mem_usage() - memory_usage;
	std::cout << " index memory: " << memory_usage << "[MB]\n\n";

        // local density computation
        compute_local_density();

        // dependency computation
        compute_dependency();

        // labeling
        labeling();
    }

    // output decision graph
    void output_decision_graph()
    {
        std::string f_name = "result/";
        if (dataset_id == 0) f_name += "0-airline";
        if (dataset_id == 1) f_name += "1-household";
        if (dataset_id == 2) f_name += "2-pamap2";
        if (dataset_id == 3) f_name += "3-sensor";
        if (dataset_id == 4) f_name += "4-tdrive";
        if (dataset_id == 5) f_name += "5-syn";

        f_name += "/dg-id(" + std::to_string(dataset_id) + ")_sampling_rate(" + std::to_string(sampling_rate) + ")_cutoff(" + std::to_string(cutoff) + ").csv";
        std::ofstream file;
        file.open(f_name.c_str(), std::ios::out | std::ios::app);

        if (file.fail())
        {
            std::cerr << " cannot open the output file." << std::endl;
            file.clear();
            return;
        }

        for (unsigned int i = 0; i < point_set.size(); ++i)
        {
            if (local_density_set[i].first > local_density_min) {
                file << i
                << "," << local_density_set[i].first
                << "," << dependent_distance_set[i]
                << "," << dependent_point[i]
                << "," << local_density_set[dependent_point[i]].first
                << "\n";
            }
        }
        file.close();
    }

    // output label
    void output_label()
    {
        std::string f_name = "result/";
        if (dataset_id == 0) f_name += "0-airline";
        if (dataset_id == 1) f_name += "1-household";
        if (dataset_id == 2) f_name += "2-pamap2";
        if (dataset_id == 3) f_name += "3-sensor";
        if (dataset_id == 4) f_name += "4-tdrive";
        if (dataset_id == 5) f_name += "5-syn";

        f_name += "/label-id(" + std::to_string(dataset_id) + ")_sampling_rate(" + std::to_string(sampling_rate) + ")_cutoff(" + std::to_string(cutoff) + ").csv";
        std::ofstream file;
        file.open(f_name.c_str(), std::ios::out | std::ios::app);

        if (file.fail())
        {
            std::cerr << " cannot open the output file." << std::endl;
            file.clear();
            return;
        }

        for (unsigned int i = 0; i < point_set.size(); ++i) file << labels[i] << "\n";
        file.close();
    }

    // result output
    void output_result()
    {
        std::string f_name = "result/";
        if (dataset_id == 0) f_name += "0-airline";
        if (dataset_id == 1) f_name += "1-household";
        if (dataset_id == 2) f_name += "2-pamap2";
        if (dataset_id == 3) f_name += "3-sensor";
        if (dataset_id == 4) f_name += "4-tdrive";
        if (dataset_id == 5) f_name += "5-syn";
        if (dataset_id == 6) f_name += "6-gas";

        f_name += "/id(" + std::to_string(dataset_id) + ")_sampling_rate(" + std::to_string(sampling_rate) + ")_cutoff(" + std::to_string(cutoff) + ")_thread_num(" + std::to_string(thread_num) + ").csv";
        std::ofstream file;
        file.open(f_name.c_str(), std::ios::out | std::ios::app);

        if (file.fail())
        {
            std::cerr << " cannot open the output file." << std::endl;
            file.clear();
            return;
        }

        file << "pre-processing time [msec]"
            << "," << "local density comp. time [msec]"
            << "," << "dependency comp. time [msec]"
            << "," << "memory usage [MB]"
            << "\n";
        file << time_vptree_build + time_offline
            << "," << time_local_density_comp
            << "," << time_dependency_comp
            << "," << memory_usage
            << "\n\n";

        file.close();        
    }
};
