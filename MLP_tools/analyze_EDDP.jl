#!/usr/bin/env julia

#=
Generate JSON or Markdown quality reports for EDDPotentials ensemble models.

Usage:

    julia --project=/path/to/EDDPotentials.jl \
        analyze_ensemble.jl CONFIG.toml OUTPUT.json [MODEL_GENERATION]

    julia --project=/path/to/EDDPotentials.jl \
        analyze_ensemble.jl CONFIG.toml OUTPUT.md [MODEL_GENERATION]

Arguments:

  CONFIG.toml       Builder calculation settings.
  OUTPUT.json|md     Output name. The extension selects JSON or Markdown.
  MODEL_GENERATION   Optional non-negative model generation. When supplied,
                     only this model's detailed section is generated.

For each model generation g, the report contains:

  1. Cumulative quality on the generation-g training snapshot (generations 0:g).
  2. Held-out quality on that snapshot's saved test split.
  3. Forward quality on structures first introduced in generations g+1:N.

Saved training/gen-g-dataset.jld2 snapshots are preferred because they already
contain computed features. If a snapshot is missing or incompatible with the
active feature definition, it is rebuilt from raw .res files in memory. The
overall section is the latest model's cumulative quality. With
MODEL_GENERATION, only that model's detailed section is generated.

All energy errors are reported in eV/atom. Spearman rank is the unweighted mean
of the per-composition rank correlations. A non-finite Spearman value is written
as null in JSON and N/A in Markdown. Structure counts reflect the filters in the
builder configuration. Raw fallback feature generation can be expensive;
--threads=auto can reduce runtime on a multicore machine.

Examples:

    # Full report for every available model generation
    julia --threads=auto --project=/path/to/EDDPotentials.jl \
        analyze_ensemble.jl link.toml ensemble-analysis.md

    # Overall report plus details for model generation 5
    julia --threads=auto --project=/path/to/EDDPotentials.jl \
        analyze_ensemble.jl link.toml ensemble-analysis-gen5.json 5

Run with --help for a short usage summary.
=#

using EDDPotentials
using JSON
using Printf
using YAML

const REPORT_SCHEMA_VERSION = 2

function print_usage(io::IO=stdout)
    script = basename(PROGRAM_FILE)
    println(io, "Usage: julia --project=/path/to/EDDPotentials.jl $script CONFIG.toml OUTPUT.{json|md} [MODEL_GENERATION]")
    println(io)
    println(io, "Analyze EDDP ensemble models against calculated structure generations.")
    println(io)
    println(io, "Arguments:")
    println(io, "  CONFIG.toml       Builder calculation settings")
    println(io, "  OUTPUT.json|md     Report path; format is selected by the extension")
    println(io, "  MODEL_GENERATION   Optional model generation for the detailed section")
end

function parse_cli(args)
    if length(args) == 1 && args[1] in ("-h", "--help")
        print_usage()
        return nothing
    end
    length(args) in (2, 3) || throw(
        ArgumentError("expected CONFIG.toml OUTPUT.{json|md} [MODEL_GENERATION]"),
    )

    config_path = abspath(args[1])
    output_path = abspath(args[2])
    isfile(config_path) || throw(ArgumentError("configuration file does not exist: $config_path"))
    splitext(config_path)[2] == ".toml" || throw(
        ArgumentError("configuration file must have a .toml extension: $config_path"),
    )

    output_format = splitext(output_path)[2]
    output_format in (".json", ".md") || throw(
        ArgumentError("output file must have a .json or .md extension: $output_path"),
    )

    generation = if length(args) == 3
        parsed = tryparse(Int, args[3])
        parsed === nothing && throw(
            ArgumentError("MODEL_GENERATION must be an integer: $(args[3])"),
        )
        parsed >= 0 || throw(ArgumentError("MODEL_GENERATION must be non-negative"))
        parsed
    else
        nothing
    end

    return (; config_path, output_path, output_format, generation)
end

function discover_model_generations(builder)
    generations = Int[]
    for name in readdir(builder.state.workdir)
        matched = match(r"ensemble-gen(\d+)\.(?:json|jld2)$", name)
        matched === nothing && continue
        generation = parse(Int, only(matched.captures))
        has_ensemble(builder, generation) && push!(generations, generation)
    end
    return sort!(unique!(generations))
end

function discover_structure_generations(builder)
    generations = Int[]
    for name in readdir(builder.state.workdir)
        matched = match(r"^gen(\d+)-dft$", name)
        matched === nothing && continue
        generation = parse(Int, only(matched.captures))
        generation <= builder.state.iteration || continue
        directory = joinpath(builder.state.workdir, name)
        isdir(directory) || continue
        any(endswith(filename, ".res") for filename in readdir(directory)) || continue
        push!(generations, generation)
    end
    return sort!(unique!(generations))
end

function discover_dataset_generations(builder)
    training_directory = joinpath(builder.state.workdir, "training")
    isdir(training_directory) || return Int[]
    generations = Int[]
    for name in readdir(training_directory)
        matched = match(r"^gen-(\d+)-dataset\.jld2$", name)
        matched === nothing && continue
        push!(generations, parse(Int, only(matched.captures)))
    end
    return sort!(unique!(generations))
end

function load_builder(config_path)
    config_directory = dirname(config_path)
    builder = cd(config_directory) do
        Builder(basename(config_path))
    end
    if !isabspath(builder.state.workdir)
        builder.state.workdir = normpath(joinpath(config_directory, builder.state.workdir))
    end
    return builder
end

function normalized_split_ratios(values)
    ratios = Float64.(values)
    total = sum(ratios)
    total > 0 || throw(ArgumentError("trainer.train_split must contain a positive entry"))
    if any(>(1), ratios) || total > 1.0 + eps(Float64)
        ratios ./= total
    end
    return ratios
end

function sidecar_label_set(builder, generation)
    yaml_path = splitext(dataset_name(builder, generation))[1] * ".yaml"
    isfile(yaml_path) || return nothing
    content = YAML.load_file(yaml_path)
    labels = Set{String}()
    for split_name in ("train", "validate", "test")
        haskey(content, split_name) || continue
        union!(labels, String.(content[split_name]))
    end
    return labels
end

function raw_generation_label_set(builder, generation)
    directory = joinpath(builder.state.workdir, "gen$(generation)-dft")
    isdir(directory) || return Set{String}()
    return Set(
        splitext(name)[1] for
        name in readdir(directory) if endswith(lowercase(name), ".res")
    )
end

function jld2_snapshot(builder, generation)
    path = dataset_name(builder, generation)
    isfile(path) || return nothing, "dataset file is missing"

    dataset_mtime = mtime(path)
    for source_generation in 0:generation
        directory = joinpath(builder.state.workdir, "gen$(source_generation)-dft")
        if isdir(directory) && mtime(directory) > dataset_mtime
            return nothing, "a source generation directory is newer than the dataset"
        end
    end

    train, validate, test = try
        load_training_dataset(builder, generation; combined=false)
    catch err
        return nothing, "dataset could not be loaded: $(sprint(showerror, err))"
    end

    stores = (train, validate, test)
    all(store -> store.feature == builder.cf, stores) ||
        return nothing, "saved feature definition differs from link.toml"
    expected_elemental_energies = make_symbol_keys(builder.state.elemental_energies)
    all(store -> store.elemental_energies == expected_elemental_energies, stores) ||
        return nothing, "saved elemental reference energies differ from link.toml"

    combined = train + validate + test
    sidecar_labels = sidecar_label_set(builder, generation)
    if sidecar_labels !== nothing && sidecar_labels != Set(combined.labels)
        return nothing, "JLD2 labels do not match the YAML sidecar"
    end

    source = Dict{String,Any}(
        "type" => "jld2_training_snapshot",
        "path" => abspath(path),
    )
    return (; train, validate, test, combined, source), nothing
end

function raw_snapshot(builder, generation, fallback_reason)
    @warn "Rebuilding generation-$generation snapshot from raw .res files" reason=fallback_reason
    structures = load_structures(builder, 0:generation)
    elemental_energies = make_symbol_keys(builder.state.elemental_energies)
    ratios = normalized_split_ratios(builder.trainer.train_split)
    train, validate, test = split_and_featurize(
        structures,
        builder.cf,
        ratios...;
        elemental_energies,
        nmax=builder.trainer.nmax,
        show_progress=true,
        use_force_data=builder.state.use_synthetic_force_data,
        nsynthetic=builder.state.synthetic_force_n,
        max_disp=builder.state.synthetic_force_max_disp,
        respect_shakes=builder.trainer.respect_shakes,
    )
    source = Dict{String,Any}(
        "type" => "raw_res_rebuilt_snapshot",
        "fallback_reason" => fallback_reason,
    )
    return (; train, validate, test, combined=train + validate + test, source)
end

function load_analysis_snapshot(builder, generation)
    @info "Loading cumulative dataset snapshot for generation $generation"
    snapshot, reason = jld2_snapshot(builder, generation)
    snapshot !== nothing && return snapshot
    GC.gc()
    return raw_snapshot(builder, generation, reason)
end

function generation_features(builder, snapshot, generation)
    labels = if snapshot.source["type"] == "jld2_training_snapshot"
        current_labels = Set(snapshot.combined.labels)
        if generation == 0
            current_labels
        else
            previous_labels = sidecar_label_set(builder, generation - 1)
            previous_labels === nothing ?
                raw_generation_label_set(builder, generation) :
                setdiff(current_labels, previous_labels)
        end
    else
        raw_generation_label_set(builder, generation)
    end

    indices = findall(label -> label in labels, snapshot.combined.labels)
    if isempty(indices)
        raw_labels = raw_generation_label_set(builder, generation)
        indices = findall(label -> label in raw_labels, snapshot.combined.labels)
    end
    isempty(indices) && error(
        "could not identify structures introduced in generation $generation within its cumulative snapshot",
    )
    return snapshot.combined[indices]
end

json_number(value::Real) = isfinite(value) ? Float64(value) : nothing

function summarize_results(results; include_max_error=false)
    formulas = [string(metadata[:formula]) for metadata in results.fc.metadata]
    summary = Dict{String,Any}(
        "number_of_structures" => length(results.fc),
        "number_of_compositions" => length(unique(formulas)),
        "energy_mae_ev_per_atom" => json_number(mae_per_atom(results)),
        "energy_rmsd_ev_per_atom" => json_number(rmse_per_atom(results)),
        "spearman_rank" => json_number(EDDPotentials.Training.spearman(results)),
    )

    if include_max_error
        absolute_errors = abs.(results.H_pred .- results.H_target) ./ results.fc.sizes
        max_error = maximum(absolute_errors)
        max_indices = findall(==(max_error), absolute_errors)
        summary["max_absolute_error_ev_per_atom"] = json_number(max_error)
        summary["max_absolute_error_structure_ids"] =
            unique(String[results.fc.labels[index] for index in max_indices])
    end

    return summary
end

function evaluate_models(
    builder,
    detail_generations,
    data_generations,
    models,
    overall_model_generation,
)
    reports = Dict{Int,Dict{String,Any}}()
    for generation in detail_generations
        reports[generation] = Dict{String,Any}(
            "model_generation" => generation,
            "cumulative_quality" => nothing,
            "test_quality" => nothing,
            "forward_generation_quality" => Any[],
        )
    end

    overall = nothing
    for data_generation in data_generations
        forward_models = [
            generation for generation in detail_generations if generation < data_generation
        ]
        needs_cumulative =
            data_generation in detail_generations ||
            data_generation == overall_model_generation
        isempty(forward_models) && !needs_cumulative && continue

        snapshot = load_analysis_snapshot(builder, data_generation)
        cumulative_summary = nothing

        if needs_cumulative
            @info "Evaluating model generation $data_generation on cumulative generations 0:$data_generation"
            cumulative_results =
                TrainingResults(models[data_generation], snapshot.combined)
            cumulative_summary =
                summarize_results(cumulative_results; include_max_error=true)
            cumulative_summary["scope"] = "cumulative_snapshot"
            cumulative_summary["structure_generations"] =
                [generation for generation in data_generations if generation <= data_generation]
            cumulative_summary["data_source"] = copy(snapshot.source)

            if data_generation in detail_generations
                reports[data_generation]["cumulative_quality"] =
                    cumulative_summary

                @info "Evaluating model generation $data_generation on its held-out test split"
                test_results =
                    TrainingResults(models[data_generation], snapshot.test)
                test_summary =
                    summarize_results(test_results; include_max_error=true)
                test_summary["scope"] = "held_out_test_split_of_cumulative_snapshot"
                test_summary["structure_generations"] =
                    [generation for generation in data_generations if generation <= data_generation]
                test_summary["data_source"] = copy(snapshot.source)
                reports[data_generation]["test_quality"] = test_summary
            end

            if data_generation == overall_model_generation
                overall = deepcopy(cumulative_summary)
                overall["model_generation"] = overall_model_generation
            end
        end

        if !isempty(forward_models)
            forward_features = generation_features(builder, snapshot, data_generation)
            for model_generation in forward_models
                @info "Evaluating model generation $model_generation on forward generation $data_generation"
                forward_results =
                    TrainingResults(models[model_generation], forward_features)
                summary = summarize_results(forward_results)
                summary["structure_generation"] = data_generation
                summary["scope"] = "structures_first_introduced_in_generation"
                source = copy(snapshot.source)
                source["selection"] =
                    "generation-$data_generation increment from the cumulative snapshot"
                summary["data_source"] = source
                push!(
                    reports[model_generation]["forward_generation_quality"],
                    summary,
                )
            end
        end

        snapshot = nothing
        cumulative_results = nothing
        test_results = nothing
        forward_results = nothing
        forward_features = nothing
        GC.gc()
    end

    for generation in detail_generations
        reports[generation]["cumulative_quality"] === nothing && error(
            "cumulative quality could not be evaluated for model generation $generation",
        )
        reports[generation]["test_quality"] === nothing && error(
            "held-out test quality could not be evaluated for model generation $generation",
        )
    end
    overall === nothing && error(
        "overall cumulative quality could not be evaluated for model generation $overall_model_generation",
    )
    return [reports[generation] for generation in detail_generations], overall
end

function build_report(config_path, requested_generation)
    builder = load_builder(config_path)
    model_generations = discover_model_generations(builder)
    structure_generations = discover_structure_generations(builder)
    dataset_generations = discover_dataset_generations(builder)
    data_generations = sort!(unique(vcat(structure_generations, dataset_generations)))
    isempty(model_generations) && error("no ensemble model files were found in $(builder.state.workdir)")
    isempty(data_generations) && error(
        "no saved datasets or calculated .res structures were found under $(builder.state.workdir)",
    )

    reportable_generations = [
        generation for generation in model_generations if generation in data_generations
    ]
    if requested_generation !== nothing
        requested_generation in reportable_generations || throw(
            ArgumentError(
                "model generation $requested_generation does not have both an ensemble and analyzable data",
            ),
        )
    end

    isempty(reportable_generations) && error(
        "no generation has both an ensemble model and analyzable data",
    )
    overall_model_generation = maximum(reportable_generations)
    detail_generations = if requested_generation === nothing
        reportable_generations
    else
        [requested_generation]
    end

    needed_generations = sort!(unique(vcat(detail_generations, [overall_model_generation])))
    models = Dict{Int,Any}()
    for generation in needed_generations
        @info "Loading ensemble model generation $generation"
        models[generation] = load_ensemble(builder, generation)
    end

    generation_reports, overall = evaluate_models(
        builder,
        detail_generations,
        data_generations,
        models,
        overall_model_generation,
    )

    return Dict{String,Any}(
        "schema_version" => REPORT_SCHEMA_VERSION,
        "configuration" => config_path,
        "available_model_generations" => model_generations,
        "available_structure_generations" => structure_generations,
        "available_dataset_generations" => dataset_generations,
        "dataset_strategy" =>
            "Prefer compatible JLD2 training snapshots; rebuild from raw .res files when unavailable.",
        "metric_definitions" => Dict{String,Any}(
            "energy_errors" => "Per-atom energy errors in eV/atom.",
            "spearman_rank" => "Unweighted mean of per-composition Spearman rank correlations; null when undefined.",
            "structure_counts" => "Counts after the data filters configured in the builder settings.",
            "cumulative_quality" => "Model generation g evaluated on the cumulative generation-g snapshot containing generations 0:g.",
            "test_quality" => "Model generation g evaluated on the held-out test split of its cumulative generation-g snapshot.",
            "forward_generation_quality" => "Model generation g evaluated on structures first introduced in each later generation.",
        ),
        "overall" => overall,
        "generations" => generation_reports,
    )
end

format_number(value) = value === nothing ? "N/A" : @sprintf("%.8g", value)

function markdown_escape(value)
    return replace(string(value), "|" => "\\|", '\n' => ' ')
end

function write_metrics_table(io, metrics; include_max_error=false)
    println(io, "| Metric | Value |")
    println(io, "|---|---:|")
    println(io, "| Number of structures | $(metrics["number_of_structures"]) |")
    println(io, "| Number of compositions | $(metrics["number_of_compositions"]) |")
    println(io, "| Energy MAE (eV/atom) | $(format_number(metrics["energy_mae_ev_per_atom"])) |")
    println(io, "| Energy RMSD (eV/atom) | $(format_number(metrics["energy_rmsd_ev_per_atom"])) |")
    println(io, "| Spearman rank | $(format_number(metrics["spearman_rank"])) |")
    if include_max_error
        println(
            io,
            "| Maximum absolute error (eV/atom) | $(format_number(metrics["max_absolute_error_ev_per_atom"])) |",
        )
    end
end

function source_description(metrics)
    source = metrics["data_source"]
    if source["type"] == "jld2_training_snapshot"
        return "JLD2 snapshot `$(source["path"])`"
    end
    reason = get(source, "fallback_reason", "saved dataset unavailable")
    return "raw .res rebuild (fallback reason: $(markdown_escape(reason)))"
end

function write_max_error_ids(io, metrics)
    println(io)
    println(io, "Maximum-AE structure ID(s):")
    println(io)
    for structure_id in metrics["max_absolute_error_structure_ids"]
        println(io, "- `$(markdown_escape(structure_id))`")
    end
end

function render_markdown(report)
    io = IOBuffer()
    overall = report["overall"]
    model_generation = overall["model_generation"]

    println(io, "# Ensemble Model Analysis")
    println(io)
    println(io, "Configuration: `$(report["configuration"])`")
    println(io)
    println(
        io,
        "Energy errors are per atom. Spearman rank is the unweighted mean of the per-composition correlations; `N/A` means it is undefined.",
    )
    println(io)
    println(io, "Dataset strategy: $(report["dataset_strategy"])")
    println(io)
    println(io, "## Overall Cumulative Quality")
    println(io)
    println(
        io,
        "Model generation **$model_generation** applied to its cumulative snapshot, structure generations **$(join(overall["structure_generations"], ", "))**.",
    )
    println(io)
    println(io, "Source: $(source_description(overall))")
    println(io)
    write_metrics_table(io, overall; include_max_error=true)
    write_max_error_ids(io, overall)

    for generation_report in report["generations"]
        generation = generation_report["model_generation"]
        cumulative = generation_report["cumulative_quality"]
        test_quality = generation_report["test_quality"]
        println(io)
        println(io, "## Model Generation $generation")
        println(io)
        println(io, "### Cumulative Quality")
        println(io)
        println(
            io,
            "Model generation **$generation** applied to cumulative structure generations **$(join(cumulative["structure_generations"], ", "))**.",
        )
        println(io)
        println(io, "Source: $(source_description(cumulative))")
        println(io)
        write_metrics_table(io, cumulative; include_max_error=true)
        write_max_error_ids(io, cumulative)

        println(io)
        println(io, "### Held-Out Test Quality")
        println(io)
        println(
            io,
            "Saved test split from the cumulative generation-$generation training snapshot.",
        )
        println(io)
        println(io, "Source: $(source_description(test_quality))")
        println(io)
        write_metrics_table(io, test_quality; include_max_error=true)
        write_max_error_ids(io, test_quality)

        following = generation_report["forward_generation_quality"]
        println(io)
        println(io, "### Forward-Generation Quality")
        println(io)
        if isempty(following)
            println(io, "No later structure generations are available.")
        else
            println(io, "| Structure generation | Source | Structures | Compositions | Energy MAE (eV/atom) | Energy RMSD (eV/atom) | Spearman rank |")
            println(io, "|---:|---|---:|---:|---:|---:|---:|")
            for metrics in following
                source_type = metrics["data_source"]["type"]
                println(
                    io,
                    "| $(metrics["structure_generation"]) | $(markdown_escape(source_type)) | $(metrics["number_of_structures"]) | $(metrics["number_of_compositions"]) | $(format_number(metrics["energy_mae_ev_per_atom"])) | $(format_number(metrics["energy_rmsd_ev_per_atom"])) | $(format_number(metrics["spearman_rank"])) |",
                )
            end
        end
    end

    return String(take!(io))
end

function render_json(report)
    io = IOBuffer()
    JSON.print(io, report, 4)
    println(io)
    return String(take!(io))
end

function main(args=ARGS)
    options = parse_cli(args)
    options === nothing && return

    report = build_report(options.config_path, options.generation)
    output = options.output_format == ".json" ? render_json(report) : render_markdown(report)
    mkpath(dirname(options.output_path))
    write(options.output_path, output)
    println("Wrote ensemble analysis to $(options.output_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch error_value
        println(stderr, "Error: ", sprint(showerror, error_value))
        exit(1)
    end
end
