using DelimitedFiles

using Statistics, DataFrames

function summarize_almlog(almlog::ALMLog, dataset_name::String)
    ssn_iters_used = almlog.alm_iter == 0 ? Int[] : almlog.ssn_iters[1:almlog.alm_iter]
    total_ssn_iters = sum(ssn_iters_used; init=0)
    total_lsearch_iters = sum(sum.(almlog.lsearch_iters[1:almlog.alm_iter]); init=0)


    df = DataFrame([(
        dataset = dataset_name,
        L_final = almlog.L_final,
        alm_time = almlog.alm_time,
        alm_iter = almlog.alm_iter,

        ssn_time_total = sum(almlog.ssn_times[1:almlog.alm_iter]),
        ssn_time_avg   = mean(almlog.ssn_times[1:almlog.alm_iter]),
        ssn_iters_total = total_ssn_iters,
        ssn_iters_avg   = mean(ssn_iters_used),

        ssn_wD_time_total = sum(almlog.ssn_wD_times[1:almlog.alm_iter]),
        ssn_wD_time_avg   = mean(almlog.ssn_wD_times[1:almlog.alm_iter]),

        prox_time_total = sum(sum.(almlog.prox_times[1:almlog.alm_iter])),
        prox_time_avg   = mean(vcat(almlog.prox_times[1:almlog.alm_iter]...)),

        prox_allocs_total = sum(almlog.prox_allocs[1:almlog.alm_iter]),
        prox_allocs_avg   = mean(almlog.prox_allocs[1:almlog.alm_iter]),

        lsearch_time_total = sum(sum.(almlog.lsearch_times[1:almlog.alm_iter])),
        lsearch_time_avg   = mean(vcat(almlog.lsearch_times[1:almlog.alm_iter]...)),

        lsearch_iters_total = total_lsearch_iters,
        lsearch_iters_avg   = total_ssn_iters == 0 ? 0.0 : total_lsearch_iters / total_ssn_iters,

        lsearch_allocs_total = sum(almlog.lsearch_allocs[1:almlog.alm_iter]),
        lsearch_allocs_avg   = mean(almlog.lsearch_allocs[1:almlog.alm_iter]),
    )])

    CSV.write("almlog_$(dataset_name).csv", df)
    return df
end


using CSV
using DataFrames

function read_all_almlog_csvs(prefix::String, N::Int; export_path::Union{Nothing, String} = nothing)
    dfs = DataFrame[]

    for i in 1:N
        filename = "$(prefix)_dataset$(i).csv"
        df = CSV.read(filename, DataFrame)
        push!(dfs, df)
    end

    full_df = vcat(dfs...)

    if export_path !== nothing
        CSV.write(export_path, full_df)
    end

    return full_df
end

