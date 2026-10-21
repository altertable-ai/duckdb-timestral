#include "timestral/forecast_grammar.hpp"

#include "timestral/forecast_aggregate.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection_manager.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/grammar_extension.hpp"
#include "duckdb/parser/peg/compiled_grammar.hpp"
#include "duckdb/parser/peg/transformer/peg_transformer.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/planner/extension_callback.hpp"

namespace duckdb {
namespace timestral {

//===--------------------------------------------------------------------===//
// Grammar
//===--------------------------------------------------------------------===//
static const char *const FORECAST_RULES[] = {
    // FROM <relation> TIMESTRAL_FORECAST (...), tried before the stock table refs. Without an alias of its own the
    // relation reads TIMESTRAL_FORECAST as its alias, hence the optional keyword.
    "ForecastTableRef <- ForecastSource ForecastKeyword? Parens(ForecastBody) TableAlias?",
    "ForecastSource <- ValuesRef / TableFunction / TableSubquery / BaseTableRef / ParensTableRef",
    "ForecastKeyword <- 'TIMESTRAL_FORECAST'",
    "ForecastBody <- Expression ForecastPartitionBy? ForecastOrderBy ForecastOption*",
    "ForecastPartitionBy <- 'PARTITION' 'BY' List(Expression)",
    "ForecastOrderBy <- 'ORDER' 'BY' OrderByExpression",
    "ForecastOption <- ForecastHorizon / ForecastEvery / ForecastModel / ForecastHistory",
    "ForecastHorizon <- 'HORIZON' Expression",
    "ForecastEvery <- 'EVERY' Expression",
    "ForecastModel <- 'USING' 'MODEL' Expression",
    "ForecastHistory <- 'WITH' 'HISTORY'",
    // timestral_forecast(value ORDER BY ts HORIZON n ...): the aggregate, spelled like the clause
    "ForecastCall <- 'TIMESTRAL_FORECAST' Parens(ForecastCallBody)",
    "ForecastCallBody <- Expression ForecastOrderBy ForecastOption*",
};

static constexpr const char *FORECAST_COLUMN = "__timestral_forecast";
static constexpr const char *IS_FORECAST_COLUMN = "is_forecast";

//===--------------------------------------------------------------------===//
// Parse result helpers
//===--------------------------------------------------------------------===//
template <class T>
using rule_transform_t = T (*)(PEGTransformer &transformer, ParseResult &parse_result);

template <class T>
static grammar_transform_process_function_t MakeTransform(rule_transform_t<T> transform) {
	return [transform](PEGTransformer &transformer, ParseResult &parse_result) {
		return transformer.MakeProcess<FinalizeTransformProcess>(
		    transformer, parse_result, [transform](PEGTransformer &inner, ParseResult &inner_result) {
			    return inner.MakeResult<T>(transform(inner, inner_result));
		    });
	};
}

static optional_ptr<ParseResult> OptionalChild(ListParseResult &list, idx_t child_idx) {
	auto &opt = list.Child<OptionalParseResult>(child_idx);
	if (!opt.HasResult()) {
		return nullptr;
	}
	return &opt.GetResult();
}

static ParseResult &SelectedChoice(ParseResult &parse_result) {
	optional_ptr<ParseResult> current = &parse_result;
	while (true) {
		if (current->type == ParseResultType::CHOICE) {
			current = &current->Cast<ChoiceParseResult>().GetResult();
			continue;
		}
		if (current->type == ParseResultType::LIST) {
			auto &list = current->Cast<ListParseResult>();
			if (list.GetChildren().size() == 1 && list.GetChild(0).type == ParseResultType::CHOICE) {
				current = &list.GetChild(0);
				continue;
			}
		}
		return *current;
	}
}

//! The children of a `Rule*` parse result (a bare Repeat, or one wrapped in a List or an Optional).
static vector<reference<ParseResult>> RepeatChildren(ParseResult &parse_result) {
	optional_ptr<ParseResult> current = &parse_result;
	if (current->type == ParseResultType::OPTIONAL) {
		auto &opt = current->Cast<OptionalParseResult>();
		if (!opt.HasResult()) {
			return {};
		}
		current = &opt.GetResult();
	}
	if (current->type == ParseResultType::LIST) {
		auto &list = current->Cast<ListParseResult>();
		if (list.GetChildren().size() == 1 && list.GetChild(0).type == ParseResultType::REPEAT) {
			current = &list.GetChild(0);
		}
	}
	vector<reference<ParseResult>> result;
	if (current->type == ParseResultType::REPEAT) {
		for (auto &child : current->Cast<RepeatParseResult>().GetChildren()) {
			result.push_back(child);
		}
	} else {
		result.push_back(*current);
	}
	return result;
}

static unique_ptr<ParsedExpression> TransformExpression(PEGTransformer &transformer, ParseResult &parse_result) {
	return transformer.Transform<unique_ptr<ParsedExpression>>(parse_result);
}

//===--------------------------------------------------------------------===//
// Specification
//===--------------------------------------------------------------------===//
struct ForecastSpec {
	unique_ptr<ParsedExpression> value;
	vector<unique_ptr<ParsedExpression>> partitions;
	unique_ptr<ParsedExpression> ts;
	unique_ptr<ParsedExpression> horizon;
	unique_ptr<ParsedExpression> every;
	unique_ptr<ParsedExpression> model;
	bool with_history = false;
};

static void SetOnce(unique_ptr<ParsedExpression> &target, unique_ptr<ParsedExpression> value, const char *keyword) {
	if (target) {
		throw ParserException("TIMESTRAL_FORECAST: %s is given more than once", keyword);
	}
	target = std::move(value);
}

// ForecastOrderBy <- 'ORDER' 'BY' OrderByExpression
static unique_ptr<ParsedExpression> TransformOrderBy(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list_pr = parse_result.Cast<ListParseResult>();
	auto order = transformer.Transform<OrderByNode>(list_pr.GetChild(2));
	if (order.type == OrderType::DESCENDING) {
		throw ParserException(
		    "TIMESTRAL_FORECAST: ORDER BY must be ascending, the forecast continues after the last time step");
	}
	return std::move(order.expression);
}

// ForecastOption*
static void TransformOptions(PEGTransformer &transformer, ParseResult &parse_result, ForecastSpec &spec) {
	for (auto &option_ref : RepeatChildren(parse_result)) {
		auto &option = SelectedChoice(option_ref.get());
		auto &list_pr = option.Cast<ListParseResult>();
		auto &name = option.Name();
		if (StringUtil::CIEquals(name, "ForecastHorizon")) {
			SetOnce(spec.horizon, TransformExpression(transformer, list_pr.GetChild(1)), "HORIZON");
		} else if (StringUtil::CIEquals(name, "ForecastEvery")) {
			SetOnce(spec.every, TransformExpression(transformer, list_pr.GetChild(1)), "EVERY");
		} else if (StringUtil::CIEquals(name, "ForecastModel")) {
			SetOnce(spec.model, TransformExpression(transformer, list_pr.GetChild(2)), "USING MODEL");
		} else if (StringUtil::CIEquals(name, "ForecastHistory")) {
			if (spec.with_history) {
				throw ParserException("TIMESTRAL_FORECAST: WITH HISTORY is given more than once");
			}
			spec.with_history = true;
		} else {
			throw InternalException("timestral: unexpected TIMESTRAL_FORECAST option rule '%s'", name);
		}
	}
	if (!spec.horizon) {
		throw ParserException("TIMESTRAL_FORECAST requires HORIZON, the number of future steps");
	}
}

//===--------------------------------------------------------------------===//
// Lowering
//===--------------------------------------------------------------------===//
//! The output name of an input expression: a column's own name, or `fallback`.
static string OutputName(const ParsedExpression &expr, const string &fallback) {
	if (expr.GetExpressionClass() == ExpressionClass::COLUMN_REF) {
		return expr.Cast<ColumnRefExpression>().GetColumnName().GetIdentifierName();
	}
	return fallback;
}

static unique_ptr<ParsedExpression> WithAlias(unique_ptr<ParsedExpression> expr, const string &alias) {
	expr->SetAlias(Identifier(alias));
	return expr;
}

//! `name := expr`, the way the parser builds it (named, and aliased for functions such as unnest)
static FunctionArgument NamedArgument(const string &name, unique_ptr<ParsedExpression> expr) {
	return FunctionArgument(Identifier(name), WithAlias(std::move(expr), name));
}

static unique_ptr<ParsedExpression> Column(const string &name) {
	return make_uniq<ColumnRefExpression>(Identifier(name));
}

//! timestral_forecast(ts, value, horizon [, {'model': ..., 'every': ...}])
static unique_ptr<ParsedExpression> ForecastAggregate(const ForecastSpec &spec) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(spec.ts->Copy());
	children.push_back(spec.value->Copy());
	children.push_back(spec.horizon->Copy());
	vector<FunctionArgument> options;
	if (spec.model) {
		options.push_back(NamedArgument("model", spec.model->Copy()));
	}
	if (spec.every) {
		options.push_back(NamedArgument("every", spec.every->Copy()));
	}
	if (!options.empty()) {
		children.push_back(make_uniq<FunctionExpression>(Identifier("struct_pack"), std::move(options)));
	}
	return make_uniq<FunctionExpression>(Identifier("timestral_forecast"), std::move(children));
}

static unique_ptr<SelectStatement> MakeStatement(unique_ptr<QueryNode> node) {
	auto statement = make_uniq<SelectStatement>();
	statement->node = std::move(node);
	return statement;
}

static vector<string> PartitionNames(const ForecastSpec &spec) {
	vector<string> names;
	for (idx_t i = 0; i < spec.partitions.size(); i++) {
		names.push_back(OutputName(*spec.partitions[i], spec.partitions[i]->ToString()));
	}
	return names;
}

//! SELECT <partitions>, unnest(f, recursive := true)
//! FROM (SELECT <partitions>, timestral_forecast(...) AS f FROM <source> GROUP BY <partitions>)
static unique_ptr<SelectNode> LowerForecastRows(const ForecastSpec &spec, unique_ptr<TableRef> source) {
	auto names = PartitionNames(spec);
	auto inner = make_uniq<SelectNode>();
	GroupingSet grouping_set;
	for (idx_t i = 0; i < spec.partitions.size(); i++) {
		inner->select_list.push_back(WithAlias(spec.partitions[i]->Copy(), names[i]));
		inner->groups.group_expressions.push_back(spec.partitions[i]->Copy());
		grouping_set.insert(ProjectionIndex(i));
	}
	if (!spec.partitions.empty()) {
		inner->groups.grouping_sets.push_back(std::move(grouping_set));
	}
	inner->select_list.push_back(WithAlias(ForecastAggregate(spec), FORECAST_COLUMN));
	inner->from_table = std::move(source);

	auto outer = make_uniq<SelectNode>();
	for (auto &name : names) {
		outer->select_list.push_back(Column(name));
	}
	vector<FunctionArgument> unnest_arguments;
	unnest_arguments.emplace_back(Column(FORECAST_COLUMN));
	unnest_arguments.push_back(NamedArgument("recursive", ConstantExpression::Boolean(true)));
	outer->select_list.push_back(make_uniq<FunctionExpression>(Identifier("unnest"), std::move(unnest_arguments)));
	outer->from_table = make_uniq<SubqueryRef>(MakeStatement(std::move(inner)));
	return outer;
}

//! The observed rows, then the forecast rows, told apart by `is_forecast`:
//! <partitions>, <ts>, <value>, forecast, q01 ... q99, is_forecast
static unique_ptr<SelectNode> LowerWithHistory(const ForecastSpec &spec, unique_ptr<TableRef> source) {
	auto names = PartitionNames(spec);
	auto ts_name = ForecastTimeField(OutputName(*spec.ts, ""));
	auto value_name = OutputName(*spec.value, "value");

	auto history = make_uniq<SelectNode>();
	for (idx_t i = 0; i < spec.partitions.size(); i++) {
		history->select_list.push_back(WithAlias(spec.partitions[i]->Copy(), names[i]));
	}
	history->select_list.push_back(WithAlias(spec.ts->Copy(), ts_name));
	history->select_list.push_back(WithAlias(spec.value->Copy(), value_name));
	history->select_list.push_back(WithAlias(ConstantExpression::Boolean(false), IS_FORECAST_COLUMN));
	history->from_table = source->Copy();

	auto forecast = LowerForecastRows(spec, std::move(source));
	forecast->select_list.push_back(WithAlias(ConstantExpression::Boolean(true), IS_FORECAST_COLUMN));

	auto both = make_uniq<SetOperationNode>();
	both->setop_type = SetOperationType::UNION_BY_NAME;
	both->setop_all = true;
	both->children.push_back(std::move(history));
	both->children.push_back(std::move(forecast));

	auto result = make_uniq<SelectNode>();
	for (auto &name : names) {
		result->select_list.push_back(Column(name));
	}
	result->select_list.push_back(Column(ts_name));
	result->select_list.push_back(Column(value_name));
	for (auto &name : ForecastValueFields()) {
		result->select_list.push_back(Column(name));
	}
	result->select_list.push_back(Column(IS_FORECAST_COLUMN));
	result->from_table = make_uniq<SubqueryRef>(MakeStatement(std::move(both)));
	return result;
}

//===--------------------------------------------------------------------===//
// Transforms
//===--------------------------------------------------------------------===//
// ForecastBody <- Expression ForecastPartitionBy? ForecastOrderBy ForecastOption*
static ForecastSpec TransformForecastBody(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list_pr = parse_result.Cast<ListParseResult>();
	ForecastSpec spec;
	spec.value = TransformExpression(transformer, list_pr.GetChild(0));
	if (auto partition_pr = OptionalChild(list_pr, 1)) {
		// ForecastPartitionBy <- 'PARTITION' 'BY' List(Expression)
		auto &partition_list = partition_pr->Cast<ListParseResult>().GetChild(2);
		for (auto &entry : PEGTransformerFactory::ExtractParseResultsFromList(partition_list)) {
			spec.partitions.push_back(TransformExpression(transformer, entry.get()));
		}
	}
	spec.ts = TransformOrderBy(transformer, list_pr.GetChild(2));
	TransformOptions(transformer, list_pr.GetChild(3), spec);
	return spec;
}

// ForecastTableRef <- ForecastSource ForecastKeyword? Parens(ForecastBody) TableAlias?
static unique_ptr<TableRef> TransformForecastTableRef(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list_pr = parse_result.Cast<ListParseResult>();
	auto source = transformer.Transform<unique_ptr<TableRef>>(SelectedChoice(list_pr.GetChild(0)));
	if (!list_pr.Child<OptionalParseResult>(1).HasResult()) {
		// `FROM t TIMESTRAL_FORECAST (...)`: the keyword was read as the alias of t
		if (!StringUtil::CIEquals(source->alias.GetIdentifierName(), "timestral_forecast") ||
		    !source->column_name_alias.empty()) {
			throw ParserException("syntax error at or near \"(\": expected TIMESTRAL_FORECAST (...) after %s",
			                      source->ToString());
		}
		source->alias = Identifier();
	}
	auto spec = TransformForecastBody(transformer, PEGTransformerFactory::ExtractResultFromParens(list_pr.GetChild(2)));
	TableAlias alias;
	transformer.TransformOptional<TableAlias>(list_pr, 3, alias);

	auto node =
	    spec.with_history ? LowerWithHistory(spec, std::move(source)) : LowerForecastRows(spec, std::move(source));
	auto result = make_uniq<SubqueryRef>(MakeStatement(std::move(node)), alias.name);
	result->column_name_alias = std::move(alias.column_name_alias);
	return std::move(result);
}

// ForecastCall <- 'TIMESTRAL_FORECAST' Parens(ForecastCallBody)
// ForecastCallBody <- Expression ForecastOrderBy ForecastOption*
static unique_ptr<ParsedExpression> TransformForecastCall(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &body = PEGTransformerFactory::ExtractResultFromParens(parse_result.Cast<ListParseResult>().GetChild(1))
	                 .Cast<ListParseResult>();
	ForecastSpec spec;
	spec.value = TransformExpression(transformer, body.GetChild(0));
	spec.ts = TransformOrderBy(transformer, body.GetChild(1));
	TransformOptions(transformer, body.GetChild(2), spec);
	if (spec.with_history) {
		throw ParserException(
		    "WITH HISTORY is only supported in FROM ... TIMESTRAL_FORECAST (...), not in timestral_forecast(...)");
	}
	return ForecastAggregate(spec);
}

//===--------------------------------------------------------------------===//
// Grammar extension
//===--------------------------------------------------------------------===//
class ForecastGrammarExtension final : public GrammarExtension {
public:
	static constexpr const char *NAME = "timestral";

	ForecastGrammarExtension()
	    : GrammarExtension(NAME, "TIMESTRAL_FORECAST clause and timestral_forecast(value ORDER BY ts HORIZON n ...)") {
	}

	vector<GrammarChange> GetChanges() const override {
		vector<GrammarChange> changes;
		for (auto rule : FORECAST_RULES) {
			changes.push_back(GrammarChange::AddRule(rule));
		}
		changes.push_back(GrammarChange::SetTransformProcess(
		    "ForecastTableRef", MakeTransform<unique_ptr<TableRef>>(TransformForecastTableRef)));
		changes.push_back(GrammarChange::SetTransformProcess(
		    "ForecastCall", MakeTransform<unique_ptr<ParsedExpression>>(TransformForecastCall)));
		changes.push_back(GrammarChange::PrependChoice("InnerTableRef", "ForecastTableRef"));
		changes.push_back(
		    GrammarChange::PrependChoice("SingleExpression", "ForecastCall", [](const PEGExpression &expression) {
			    return expression.type == PEGExpression::Type::REFERENCE &&
			           StringUtil::CIEquals(expression.text.GetString(), "FunctionExpression");
		    }));
		return changes;
	}

	//! DuckDB's default grammar extended with only this extension (compiled once, shared by connections).
	shared_ptr<CompiledGrammar> GetCompiledGrammar() {
		lock_guard<mutex> guard(grammar_lock);
		if (!compiled_grammar) {
			vector<reference<GrammarExtension>> extensions;
			extensions.emplace_back(*this);
			compiled_grammar = CompiledGrammar::Create(extensions);
		}
		return compiled_grammar;
	}

	//! Adds this extension to the connection's active grammar extensions.
	void Activate(ClientContext &context) {
		auto &config = ClientConfig::GetConfig(context);
		auto &active = config.active_grammar_extensions;
		for (auto &name : active) {
			if (StringUtil::CIEquals(name, NAME)) {
				return;
			}
		}
		active.emplace_back(NAME);
		// Same effect as appending to `SET active_grammar_extensions`, without recompiling the common case.
		config.cached_grammar = active.size() == 1 ? GetCompiledGrammar() : CompiledGrammar::Create(context, active);
	}

private:
	mutex grammar_lock;
	shared_ptr<CompiledGrammar> compiled_grammar;
};

//! Connections opened after LOAD parse with the grammar from the start.
class ForecastGrammarActivation final : public ExtensionCallback {
public:
	explicit ForecastGrammarActivation(shared_ptr<ForecastGrammarExtension> grammar_p) : grammar(std::move(grammar_p)) {
	}

	void OnConnectionOpened(ClientContext &context) override {
		grammar->Activate(context);
	}

private:
	shared_ptr<ForecastGrammarExtension> grammar;
};

//! Connections already open at LOAD (including the one running it) switch on their own thread once their current or
//! next query ends, since another thread may be parsing with their grammar right now.
class PendingForecastGrammarActivation final : public ClientContextState {
public:
	static constexpr const char *KEY = "timestral_pending_grammar_activation";

	explicit PendingForecastGrammarActivation(shared_ptr<ForecastGrammarExtension> grammar_p)
	    : grammar(std::move(grammar_p)) {
	}

	void QueryEnd(ClientContext &context) override {
		grammar->Activate(context);
		// One-shot: a later `SET active_grammar_extensions` must stick.
		context.registered_state->Remove(KEY);
	}

private:
	shared_ptr<ForecastGrammarExtension> grammar;
};

void RegisterForecastGrammar(DatabaseInstance &db) {
	auto grammar = make_shared_ptr<ForecastGrammarExtension>();
	GrammarExtension::Register(db, grammar);
	ExtensionCallback::Register(DBConfig::GetConfig(db), make_shared_ptr<ForecastGrammarActivation>(grammar));
	for (auto &connection : db.GetConnectionManager().GetConnectionList()) {
		connection->registered_state->GetOrCreate<PendingForecastGrammarActivation>(
		    PendingForecastGrammarActivation::KEY, grammar);
	}
}

} // namespace timestral
} // namespace duckdb
