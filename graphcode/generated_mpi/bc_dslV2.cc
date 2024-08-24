#include "bc_dslV2.h"

void Compute_BC(Graph& g, NodeProperty<float>& BC, std::set<int>& sourceSet, boost::mpi::communicator world) {
    BC.attachToGraph(&g, 0.0f);
    NodeProperty<float> sigma;
    NodeProperty<float> delta;
    std::set<int>::iterator itr;

    for (itr = sourceSet.begin(); itr != sourceSet.end(); itr++) {
        int src = *itr;
        delta.attachToGraph(&g, 0.0f);
        sigma.attachToGraph(&g, 0.0f);
        sigma.setValue(src, 1.0f);
        g.create_bfs_dag(src);

        for (int phase = 0; phase < g.num_bfs_phases() - 1; phase++) {
            for (int v : g.get_bfs_nodes_for_phase(phase)) {
//                std::cout << "Process " << world.rank() << " phase" << phase << " vertex " << v << std::endl;
                for (int w : g.get_bfs_children(v)) {
                    // Simplified condition: assuming we check if w should belong to next phase
                    sigma.atomicAdd(w, sigma.getValue(v));
                }
            }
            sigma.fatBarrier();
            world.barrier();
        }

//        // Check sigma values
//        for (int i = 0; i < g.num_nodes(); i++) {
//            std::cout << "Sigma: " << i << " " << sigma.getValue(i) << std::endl;
//        }

        for (int phase = g.num_bfs_phases() - 2; phase > 0; phase--) {
            for (int v : g.get_bfs_nodes_for_phase(phase)) {
                for (int w : g.get_bfs_children(v)) {
                    float contrib = (sigma.getValue(v) / sigma.getValue(w)) * (1.0f + delta.getValue(w));
                    delta.atomicAdd(v, contrib);
//                    std::cout << "v " << v << " w " << w << " contrib" << contrib << " new_delta: " << delta.getValue(v) << std::endl;
                }
                delta.fatBarrier();
                BC.atomicAdd(v, delta.getValue(v));
            }
            BC.fatBarrier();
            world.barrier();
        }
        world.barrier();
    }
}
