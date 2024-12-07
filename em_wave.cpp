#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>  // 添加这行
// #include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>

// 添加 Eigen 头文件
#include <Eigen/Sparse>
#include <Eigen/SparseLU>  // 或者使用其他 Eigen 求解器

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <complex>

namespace EMWave
{
using namespace dealii;

// PML参数设置
struct PMLParameters
{
    double sigma_max = 5.0;   // 减小最大衰减系数
    double pml_width = 0.25;  // 调整 PML 层厚度
    
    double get_sigma(const Point<2> &p) const
    {
        double x = std::abs(p[0]);
        double y = std::abs(p[1]);
        double sigma_x = 0.0;
        double sigma_y = 0.0;
        
        const double x_start = 1.0 - pml_width;
        const double y_start = 1.0 - pml_width;
        
        if (x > x_start)
        {
            double x_norm = (x - x_start)/pml_width;
            sigma_x = sigma_max * x_norm * x_norm;  // 使用二次函数而不是三次函数
        }
        
        if (y > y_start)
        {
            double y_norm = (y - y_start)/pml_width;
            sigma_y = sigma_max * y_norm * y_norm;
        }
            
        return sigma_x + sigma_y;
    }
};

template <int dim>
class EMWaveProblem
{
public:
    EMWaveProblem();
    void run();

private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void output_results() const;
    void output_grid() const;

    Triangulation<dim> triangulation;
    FE_Q<dim>         fe;
    DoFHandler<dim>   dof_handler;

    AffineConstraints<std::complex<double>> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<std::complex<double>> system_matrix;
    Vector<std::complex<double>>       solution;
    Vector<std::complex<double>>       system_rhs;

    PMLParameters pml_params;
};

template <int dim>
EMWaveProblem<dim>::EMWaveProblem()
    : fe(1)
    , dof_handler(triangulation)
{}

template <int dim>
void EMWaveProblem<dim>::make_grid()
{
    GridGenerator::hyper_cube(triangulation, -1, 1);
    
    // 在PML区域使用更细的网格
    const unsigned int initial_refinements = 8;
    triangulation.refine_global(initial_refinements);
    
    // 对PML区域进行额外的局部加密
    for (const auto &cell : triangulation.active_cell_iterators())
    {
        const Point<dim> cell_center = cell->center();
        if (std::abs(cell_center[0]) > 0.7 || std::abs(cell_center[1]) > 0.7)
        {
            cell->set_refine_flag();
        }
    }
    triangulation.execute_coarsening_and_refinement();
    
    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;
              
    output_grid();
}

template <int dim>
void EMWaveProblem<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    
    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;
    
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim, std::complex<double>>(),
                                           constraints);
    constraints.close();
    
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                   dsp,
                                   constraints,
                                   /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void EMWaveProblem<dim>::assemble_system()
{
    QGauss<dim> quadrature_formula(fe.degree + 2);
    
    FEValues<dim> fe_values(fe,
                           quadrature_formula,
                           update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<std::complex<double>> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<std::complex<double>>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const double omega = 5.0;  // 角频率
    const std::complex<double> i(0.0, 1.0);
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs    = 0;

        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
            const auto x_q = fe_values.quadrature_point(q_index);
            const double sigma = pml_params.get_sigma(x_q);
            std::complex<double> s_factor = 1.0 - i * sigma/omega;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_matrix(i, j) +=
                        ((1.0/s_factor * fe_values.shape_grad(i, q_index) *
                          fe_values.shape_grad(j, q_index) -
                          omega * omega * s_factor *
                          fe_values.shape_value(i, q_index) *
                          fe_values.shape_value(j, q_index))) *
                        fe_values.JxW(q_index);
                }

                // 添加高斯源
                const double source_width = 0.1;
                cell_rhs(i) += 
                    std::exp(-x_q.square()/(2.0*source_width*source_width)) *
                    fe_values.shape_value(i, q_index) *
                    fe_values.JxW(q_index);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                            local_dof_indices,
                                            system_matrix, system_rhs);
    }
}


template <int dim>
void EMWaveProblem<dim>::solve()
{
    std::cout << "Solving linear system using Eigen..." << std::endl;
    std::cout << "Matrix size: " << system_matrix.m() 
              << " x " << system_matrix.n() << std::endl;
              
    try {
        // 将 deal.II 矩阵转换为 Eigen 格式
        const unsigned int n = system_matrix.m();
        Eigen::SparseMatrix<std::complex<double>> eigen_matrix(n, n);
        std::vector<Eigen::Triplet<std::complex<double>>> triplets;
        
        // 预留空间以提高效率
        triplets.reserve(system_matrix.n_nonzero_elements());
        
        // 转换矩阵格式
        for (unsigned int row = 0; row < n; ++row)
        {
            for (auto it = system_matrix.begin(row); it != system_matrix.end(row); ++it)
            {
                triplets.push_back(Eigen::Triplet<std::complex<double>>(
                    row, it->column(), it->value()));
            }
        }
        eigen_matrix.setFromTriplets(triplets.begin(), triplets.end());
        
        // 转换右端向量
        Eigen::VectorXcd eigen_rhs(n);
        for (unsigned int i = 0; i < n; ++i)
        {
            eigen_rhs(i) = system_rhs(i);
        }
        
        // 使用 Eigen 求解器
        Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> solver;
        // 也可以尝试其他求解器，如：
        // Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>>> solver;
        // Eigen::GMRES<Eigen::SparseMatrix<std::complex<double>>> solver;
        
        solver.compute(eigen_matrix);
        if(solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigen decomposition failed");
        }
        
        Eigen::VectorXcd eigen_solution = solver.solve(eigen_rhs);
        if(solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigen solver failed");
        }
        
        // 将结果转回 deal.II 格式
        for (unsigned int i = 0; i < n; ++i)
        {
            solution(i) = eigen_solution(i);
        }
        
        constraints.distribute(solution);
        
        std::cout << "Eigen solver completed successfully." << std::endl;
    }
    catch (std::exception &exc) {
        std::cerr << exc.what() << std::endl;
        throw;
    }
}



template <int dim>
void EMWaveProblem<dim>::output_grid() const
{
    GridOut grid_out;
    std::ofstream grid_file("grid.eps");
    grid_out.write_eps(triangulation, grid_file);
}


template <int dim>
void EMWaveProblem<dim>::output_results() const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    
    Vector<double> real_solution(solution.size());
    Vector<double> imag_solution(solution.size());
    Vector<double> amplitude(solution.size());
    
    for(unsigned int i = 0; i < solution.size(); ++i)
    {
        real_solution[i] = solution[i].real();
        imag_solution[i] = solution[i].imag();
        amplitude[i] = std::abs(solution[i]);
    }
    
    data_out.add_data_vector(real_solution, "real_part");
    data_out.add_data_vector(imag_solution, "imag_part");
    data_out.add_data_vector(amplitude, "amplitude");
    
    data_out.build_patches();
    
    // 输出VTK格式以便使用ParaView等工具可视化
    std::ofstream output("solution.vtk");
    data_out.write_vtk(output);
    
    // 也输出SVG格式以便在浏览器中查看
    std::ofstream output_svg("solution.svg");
    data_out.write_svg(output_svg);
}

template <int dim>
void EMWaveProblem<dim>::run()
{
    std::cout << "Solving EM wave problem with PML" << std::endl;
    
    make_grid();
    setup_system();
    assemble_system();
    solve();
    output_results();
    
    std::cout << "Results have been written to solution.vtk and solution.svg" << std::endl;
}

} // namespace EMWave

int main()
{
    try
    {
        std::cout << "Starting EM wave simulation..." << std::endl;
        EMWave::EMWaveProblem<2> em_wave_problem;
        em_wave_problem.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}