#include "covertree.h"
#include <chrono>
#include <queue>
#include <cfloat>


// vp-tree build time
double time_vptree_build = 0;
std::chrono::system_clock::time_point  _s, _e;


// declaration of "node"
class node;

// implementation of "node"
class node {

public:

	// identifier of vantage point
	int identifier = -1;

	// mean distane
	float distance_mean = 0;

	// pointer to parent
	node* parent = NULL;

	// pointer to left node
	node* left = NULL;

	// pointer to right node
	node* right = NULL;

	// set of data object identifier (only for leaf node)
	std::vector<unsigned int> leaf;

	// sub-tree size
	unsigned int cnt = 0;


	/***************/
	/* constructor */
	/***************/

	// standard
	node() {}

	// with identifier
	node(const unsigned int id) { identifier = id; }


	/*******************/
	/* member function */
	/*******************/
	void update_id(const unsigned int id) { identifier = id; }
};


class vp_tree {

	std::vector<node> vptree;
	std::vector<node*> leave;
    std::vector<pointType> point_set;
    std::vector<std::pair<std::vector<float>, unsigned int>> pset;

    unsigned int dim = 0;

    // capacity of one node
    const unsigned int capacity = 4;

public:
	/***************/
	/* constructor */
	/***************/
	vp_tree() {}


private:
	/*******************/
	/* member function */
	/*******************/

    // distance computation
    double compute_distance(const pointType& l, const pointType& r) {

        float distance = 0;
        for (unsigned int i = 0; i < dim; ++i) distance += (l.first[i] - r.first[i]) * (l.first[i] - r.first[i]);
        return sqrt(distance);
    }

    double compute_distance(const std::vector<float>& l, const std::vector<float>& r) {

        float distance = 0;
        for (unsigned int i = 0; i < dim; ++i) distance += (l[i] - r[i]) * (l[i] - r[i]);
        return sqrt(distance);
    }


	// select vantage point randomly
	unsigned int select_vp(const unsigned int size) {

		// uniform distribution
		std::mt19937 mt(0);
		std::uniform_int_distribution<> rnd(0, (int)size - 1);
		unsigned int idx = (unsigned int)rnd(mt);

		return idx;
	}

	// split
	void split(node* n, const std::vector<unsigned int>& idx_set) {

		const unsigned int size = idx_set.size();

		if (size > capacity) {

			// decide vp
			unsigned int idx = select_vp(size);

			// update node
			n->update_id(idx_set[idx]);

			n->cnt = size;

			// compute distance mean
			std::vector<float> distances;
			distances.resize(size);

            for (unsigned int i = 0; i < size; ++i) distances[i] = compute_distance(pset[n->identifier].first, pset[idx_set[i]].first);
			std::vector<float> distances_cpy = distances;
			std::sort(distances_cpy.begin(), distances_cpy.end());

			n->distance_mean = distances_cpy[distances_cpy.size() / 2];

			// make new subsets
			std::vector<unsigned int> idx_set_l, idx_set_r;
			for (unsigned int i = 0; i < size; ++i) {

				if (i != idx) {
					if (distances[i] <= n->distance_mean) {
						idx_set_l.push_back(idx_set[i]);
					}
					else {
						idx_set_r.push_back(idx_set[i]);
					}
				}
			}

			if (idx_set_r.size() > 0) {

				// make nodes
				node left, right;
				vptree.push_back(left);
				vptree.push_back(right);

				// update children
				node* l = &vptree[vptree.size() - 2];
				node* r = &vptree[vptree.size() - 1];

				// set children
				n->left = l;
				n->right = r;

				// set parent
				l->parent = n;
				r->parent = n;

				split(l, idx_set_l);
				split(r, idx_set_r);
			}
			else {

				// leaf node
				n->identifier = -1;
				n->leaf = idx_set;
			}
		}
		else {

			// leaf node
			n->leaf = idx_set;
			n->cnt = size;
		}
	}

public:

    // input
    void input(const std::vector<pointType>& _point_set, const unsigned int dimensionality) { 
        point_set = _point_set;
        dim = dimensionality;
    }

    void input(const std::vector<std::vector<float>>& _point_set, const unsigned int dimensionality) { 
        for (unsigned int i = 0; i < _point_set.size(); ++i) pset.push_back({_point_set[i], i});
        dim = dimensionality;
    }

	void input(const std::vector<std::pair<std::vector<float>, unsigned int>>& _point_set, const unsigned int dimensionality) { 
        pset = _point_set;
        dim = dimensionality;
    }

	// build a vp-tree
	double build() {

		_s = std::chrono::system_clock::now();

        //const unsigned int size = point_set.size();
        const unsigned int size = pset.size();

		// make a set of identifier
		std::vector<unsigned int> idx_set;
		for (unsigned int i = 0; i < size; ++i) idx_set.push_back(i);

		// reserve memory for vp-tree
		vptree.reserve(size);

		// make the root node
		node root;
		vptree.push_back(root);

		// recursive split
		split(&vptree[0], idx_set);

		_e = std::chrono::system_clock::now();
		time_vptree_build = std::chrono::duration_cast<std::chrono::microseconds>(_e - _s).count();
        time_vptree_build /= 1000;
		return time_vptree_build;
	}

	// range search
    unsigned int range_count(const std::vector<float>& query, const unsigned int idx, const float cutoff)
	{
		unsigned int result_size = 0;

		// prepare a queue
		std::deque<node*> queue;

		// insert root into the queue
		queue.push_back(&vptree[0]);

		// recursive search
		while (queue.size() > 0) {

			// pop the front
			node* n = queue[0];
			queue.pop_front();

			if (n->identifier == -1) {

				// leaf node
				for (unsigned int i = 0; i < n->leaf.size(); ++i)
				{
					const float distance = compute_distance(query, pset[n->leaf[i]].first);
					if (distance < cutoff) ++result_size;
				}
			}
			else {

				// intermediate node

				// distance computation between query and vp
                const float distance = compute_distance(query, pset[n->identifier].first);
				if (distance < cutoff) ++result_size;

				// children check
				if (distance + cutoff >= n->distance_mean) queue.push_back(n->right);
				if (distance - cutoff <= n->distance_mean) {
					if (distance + n->distance_mean < cutoff) {
						result_size += n->left->cnt;
					}
					else {
						queue.push_back(n->left);
					}
				}
			}
		}

		return result_size;
	}

	// k-NN search
	void knn_search(const std::vector<float>& query, const unsigned int k, std::vector<unsigned int> &knn_idx) {

		//_s = std::chrono::system_clock::now();

		std::map<float, unsigned int> result;
		float threshold = FLT_MAX;

		// prepare a queue
		std::deque<node*> queue;

		// insert root into the queue
		queue.push_back(&vptree[0]);

		// recursive search
		while (queue.size() > 0) {

			// pop the front
			node* n = queue[0];
			queue.pop_front();

			if (n->identifier == -1) {

				// leaf node
				for (unsigned int i = 0; i < n->leaf.size(); ++i) {

					// get idx
					const unsigned int idx = n->leaf[i];

					const float distance = compute_distance(query, pset[idx].first);
					if (distance < threshold) {
						result.insert({distance, pset[idx].second});
						if (result.size() > k) {
							auto it = result.end();
							--it;
							result.erase(it);
						}
						if (result.size() == k) {
							auto it = result.end();
							--it;
							threshold = it->first;
						}
					}
				}
			}
			else {

				// intermediate node

				// distance computation between query and vp
				const float distance = compute_distance(query, pset[n->identifier].first);

				if (distance < threshold) {
					result.insert({distance, pset[n->identifier].second});
					if (result.size() > k) {
						auto it = result.end();
						--it;
						result.erase(it);
					}
					if (result.size() == k) {
						auto it = result.end();
						--it;
						threshold = it->first;
					}
				}

				// check children
				node* c = n->left;
				if (distance - threshold <= n->distance_mean) queue.push_back(n->left);
				if (distance + threshold >= n->distance_mean) queue.push_back(n->right);
			}
		}

		unsigned int cnt = 0;
		auto it = result.begin();
		while (it != result.end()) {
			knn_idx[cnt] = it->second;
			++it;
			++cnt;
		}

		//_e = std::chrono::system_clock::now();
		//double time_knn = std::chrono::duration_cast<std::chrono::microseconds>(_e - _s).count();
        //time_knn /= 1000;
		//std::cout << " k-NNS time: " << time_knn << "[msec]\n";
	}

	// NN search
	std::pair<float, unsigned int> nn_search(const std::vector<float>& query) {

		unsigned int _idx = 0;
		float threshold = FLT_MAX;

		// prepare a queue
		std::deque<node*> queue;

		// insert root into the queue
		queue.push_back(&vptree[0]);

		// recursive search
		while (queue.size() > 0) {

			// pop the front
			node* n = queue[0];
			queue.pop_front();

			if (n->identifier == -1) {

				// leaf node
				for (unsigned int i = 0; i < n->leaf.size(); ++i) {

					// get idx
					const unsigned int idx = n->leaf[i];

					const float distance = compute_distance(query, pset[idx].first);
					if (distance < threshold) {
						threshold = distance;
						_idx = pset[idx].second;
					}
				}
			}
			else {

				// intermediate node

				// distance computation between query and vp
				const float distance = compute_distance(query, pset[n->identifier].first);

				if (distance < threshold) {
					threshold = distance;
					_idx = pset[n->identifier].second;
				}

				// check children
				node* c = n->left;
				if (distance - threshold <= n->distance_mean) queue.push_back(n->left);
				if (distance + threshold >= n->distance_mean) queue.push_back(n->right);
			}
		}

		return {threshold, _idx};
	}

};