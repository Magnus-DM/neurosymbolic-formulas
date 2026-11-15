# Based on the AND-OR tree evaluator provided by the course Learning and Reasoning
# (KU Leuven, 2025-2026). Rewritten by Magnus De Maerteleire to evaluate each neural
# predicate once per batch instead of once per tree node.

import torch

from nesy.term import Term


class Evaluator:
    def __init__(self, label_semantics, neural_predicates):
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics

    def evaluate(self, tensor_sources, and_or_trees, queries):
        device = None
        for src in tensor_sources.values():
            if isinstance(src, torch.Tensor):
                device = src.device
                break
            if hasattr(src, "__len__") and len(src) > 0 and isinstance(src[0], torch.Tensor):
                device = src[0].device
                break
        if device is None:
            device = torch.device("cpu")

        one = torch.tensor(1.0, device=device)
        nn_parsed = {}

        def term_key(t):
            try:
                hash(t)
                return t
            except TypeError:
                return id(t)

        def parse_nn(val: Term):
            k = term_key(val)
            if k in nn_parsed:
                return nn_parsed[k]

            # Neural predicate structure: nn(id_neural_predicate, input1, input2, ..., output)
            id_pred = val.arguments[0].functor
            pred = self.neural_predicates[id_pred]

            out_functor = val.arguments[-1].functor
            output_idx = pred.output_encoding[out_functor]

            inputs_spec = []
            for arg in val.arguments[1:-1]:
                tensor_source = arg.arguments[0].functor
                tensor_id = int(arg.arguments[1].functor)
                inputs_spec.append((tensor_source, tensor_id))
            inputs_spec = tuple(inputs_spec)

            nn_parsed[k] = (id_pred, pred, output_idx, inputs_spec)
            return nn_parsed[k]

        # collect all nn-terms in subtree
        def collect_nn_terms(root_node):
            seen_nodes = set()
            nn_terms = []

            stack = [root_node]
            while stack:
                node = stack.pop()
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)

                val = and_or_trees[node]
                if val is None:
                    continue
                elif isinstance(val, set):
                    for conj in val:
                        for child in conj:
                            stack.append(child)
                elif isinstance(val, Term) and val.functor == "nn":
                    nn_terms.append(val)
                else:
                    raise Exception(f"Cannot evaluate node {node}")

            uniq = []
            seen_terms = set()
            for t in nn_terms:
                k = term_key(t)
                if k not in seen_terms:
                    seen_terms.add(k)
                    uniq.append(t)
            return uniq

        # batched NN-evaluation for one i
        def eval_all_nn_terms_batched(nn_terms, i: int):
            """
            Retourneert dict: term_key(term) -> scalar tensor result
            Batcht per (id_pred, inputs_spec).
            """
            results = {}

            # group by (id_pred, inputs_spec) so that batching is safe
            buckets = {}
            for t in nn_terms:
                id_pred, pred, output_idx, inputs_spec = parse_nn(t)
                buckets.setdefault((id_pred, inputs_spec), []).append((t, pred, output_idx))

            for (id_pred, inputs_spec), items in buckets.items():
                pred = items[0][1]

                # Bouw batch input: [B, num_inputs, ...]
                batch_inputs = []
                for t, _, _ in items:
                    # inputs_spec is identiek binnen bucket
                    inputs = [tensor_sources[src][i][tid] for (src, tid) in inputs_spec]
                    batch_inputs.append(torch.stack(inputs, dim=0))
                batch_inputs = torch.stack(batch_inputs, dim=0)

                # Probeer batched forward
                try:
                    out = pred(batch_inputs)  # verwacht [B, C]
                    for b, (t, _, output_idx) in enumerate(items):
                        results[term_key(t)] = out[b, output_idx]
                    continue
                except Exception:
                    # fallback: single forward per term
                    pass

                for t, _, output_idx in items:
                    inputs = [tensor_sources[src][i][tid] for (src, tid) in inputs_spec]
                    input_tensor = torch.stack(inputs, dim=0)
                    out = pred(input_tensor)
                    results[term_key(t)] = out[0, output_idx]

            return results

        # evaluate the tree with memoization, use NN-cache
        def eval_tree(root_node, i: int, nn_values):
            cache_node = {}  # node -> tensor

            def eval_node(node):
                if node in cache_node:
                    return cache_node[node]

                val = and_or_trees[node]
                if val is None:
                    res = one

                elif isinstance(val, set):
                    if len(val) == 0:
                        raise Exception(f"Cannot find {node} in and-or trees")
                    vals = [eval_conj(conj) for conj in val]
                    res = self.label_semantics.reduce_disjunction(torch.stack(vals, dim=-1), dim=-1)

                elif isinstance(val, Term) and val.functor == "nn":
                    k = term_key(val)
                    res = nn_values[k]

                else:
                    raise Exception(f"Cannot evaluate node {node}")

                cache_node[node] = res
                return res

            def eval_conj(conj):
                vals = [eval_node(c) for c in conj]
                return self.label_semantics.reduce_conjunction(torch.stack(vals, dim=-1), dim=-1)

            return eval_node(root_node)

        all_nn_terms = []
        query_nn_terms = []
        seen_nn_terms = set()

        first_query_nn_terms = collect_nn_terms(queries[0]) if queries else []
        all_queries_same_nn_terms = True
        for query in queries:
            nn_terms = collect_nn_terms(query)
            query_nn_terms.append(nn_terms)
            # Check if this query has the same NN terms as the first
            if len(nn_terms) != len(first_query_nn_terms):
                all_queries_same_nn_terms = False
            else:
                first_keys = {term_key(t) for t in first_query_nn_terms}
                this_keys = {term_key(t) for t in nn_terms}
                if first_keys != this_keys:
                    all_queries_same_nn_terms = False

            # Add unique terms to all_nn_terms
            for t in nn_terms:
                k = term_key(t)
                if k not in seen_nn_terms:
                    seen_nn_terms.add(k)
                    all_nn_terms.append(t)

        # If all queries use the same NN terms, we can reuse the first query's terms
        if all_queries_same_nn_terms and len(queries) > 1:
            # Reuse the same nn_terms list for all queries
            for i in range(1, len(queries)):
                query_nn_terms[i] = query_nn_terms[0]

        # Determine batch size from tensor_sources
        batch_size = None
        for src in tensor_sources.values():
            if isinstance(src, torch.Tensor):
                batch_size = src.shape[0]
                break
            elif hasattr(src, "__len__") and len(src) > 0:
                if isinstance(src[0], torch.Tensor):
                    batch_size = src[0].shape[0]
                    break
                elif hasattr(src[0], "__len__") and len(src[0]) > 0:
                    batch_size = len(src[0])
                    break

        if batch_size is None:
            batch_size = len(queries)

        # Evaluate all unique NN terms for all samples in batch at once
        all_nn_values = {}

        if len(all_nn_terms) > 0:

            def get_tensor(src_name, sample_idx, tensor_idx):
                src = tensor_sources[src_name]
                if isinstance(src, torch.Tensor):
                    # LogicInArchitecture: src is (B, K, ...), src[i] is (K, ...)
                    return src[sample_idx][tensor_idx]
                else:
                    # LogicInLoss: src is list of tensors, src[i] is (K, ...)
                    return src[sample_idx][tensor_idx]

            # Group by (id_pred, inputs_spec) for batching
            buckets = {}
            for t in all_nn_terms:
                id_pred, pred, output_idx, inputs_spec = parse_nn(t)
                buckets.setdefault((id_pred, inputs_spec), []).append((t, pred, output_idx))

            # Evaluate each bucket for all samples
            for (id_pred, inputs_spec), items in buckets.items():
                pred = items[0][1]

                try:
                    # Collect all inputs: [batch_size * num_items, num_inputs, ...]
                    all_inputs = []
                    for i in range(batch_size):
                        for t, _, _ in items:
                            inputs = [get_tensor(src, i, tid) for (src, tid) in inputs_spec]
                            all_inputs.append(torch.stack(inputs, dim=0))

                    batch_inputs = torch.stack(all_inputs, dim=0)

                    out = pred(batch_inputs)  # [batch_size * num_items, C]
                    out = out.view(batch_size, len(items), -1)  # [batch_size, num_items, C]

                    # Store results
                    for item_idx, (t, _, output_idx) in enumerate(items):
                        k = term_key(t)
                        for i in range(batch_size):
                            all_nn_values[(k, i)] = out[i, item_idx, output_idx]
                    continue
                except Exception:
                    # Fallback: evaluate per sample, but still batch items within sample
                    pass

                # Fallback: evaluate per sample, but batch items within sample
                for i in range(batch_size):
                    sample_inputs = []
                    for t, _, _ in items:
                        inputs = [get_tensor(src, i, tid) for (src, tid) in inputs_spec]
                        sample_inputs.append(torch.stack(inputs, dim=0))

                    try:
                        batch_inputs = torch.stack(sample_inputs, dim=0)
                        out = pred(batch_inputs)  # [num_items, C]
                        for item_idx, (t, _, output_idx) in enumerate(items):
                            k = term_key(t)
                            all_nn_values[(k, i)] = out[item_idx, output_idx]
                    except Exception:
                        # Final fallback: single forward per term
                        for t, _, output_idx in items:
                            inputs = [get_tensor(src, i, tid) for (src, tid) in inputs_spec]
                            input_tensor = torch.stack(inputs, dim=0)
                            out = pred(input_tensor)
                            k = term_key(t)
                            all_nn_values[(k, i)] = out[0, output_idx]

        # Now evaluate all queries using pre-computed NN values
        out_vals = []
        for i, query in enumerate(queries):
            # Build nn_values dict for this query and sample
            nn_values = {}
            for t in query_nn_terms[i]:
                k = term_key(t)
                # For training: i corresponds to sample index
                sample_idx = i if i < batch_size else i % batch_size
                if (k, sample_idx) in all_nn_values:
                    nn_values[k] = all_nn_values[(k, sample_idx)]
                else:
                    nn_terms_single = [t]
                    nn_values_single = eval_all_nn_terms_batched(nn_terms_single, sample_idx)
                    nn_values.update(nn_values_single)

            out_vals.append(eval_tree(query, i, nn_values))

        return torch.stack(out_vals, dim=0)
