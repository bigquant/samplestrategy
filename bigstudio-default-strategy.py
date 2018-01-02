# 本代码由可视化策略环境自动生成  
# 本代码单元只能在可视化模式下编辑。您也可以拷贝代码，粘贴到新建的代码单元或者策略，然后修改。


m1 = M.instruments.v2(
    start_date='2010-01-01',
    end_date='2015-01-01',
    market='CN_STOCK_A',
    instrument_list='',
    max_count=0
)

m2 = M.advanced_auto_labeler.v2(
    instruments=m1.data,
    label_expr="""# #号开始的表示注释
# 0. 每行一个，顺序执行，从第二个开始，可以使用label字段
# 1. 可用数据字段见 https://bigquant.com/docs/data_history_data.html
#   添加benchmark_前缀，可使用对应的benchmark数据
# 2. 可用操作符和函数见 `表达式引擎 <https://bigquant.com/docs/big_expr.html>`_

# 计算收益：5日收盘价(作为卖出价格)除以明日开盘价(作为买入价格)
shift(close, -2) / shift(open, -1)

# 极值处理：用1%和99%分位的值做clip
clip(label, all_quantile(label, 0.01), all_quantile(label, 0.99))

# 将分数映射到分类，这里使用20个分类
all_wbins(label, 20)

# 过滤掉一字涨停的情况 (设置label为NaN，在后续处理和训练中会忽略NaN的label)
where(shift(high, -1) == shift(low, -1), NaN, label)
""",
    start_date='',
    end_date='',
    benchmark='000300.SHA',
    drop_na_label=True,
    cast_label_int=True
)

m3 = M.input_features.v1(
    features="""mf_net_amount_xl_0
mf_net_pct_xl_0
mf_net_amount_l_0
mf_net_pct_l_0
sum(mf_net_amount_xl_0, 5)
sum(mf_net_amount_l_0, 5)
volatility_5_0
market_cap_float_0
 
 """
)

m4 = M.general_feature_extractor.v6(
    instruments=m1.data,
    features=m3.data,
    start_date='',
    end_date='',
    before_start_days=0
)

m5 = M.derived_feature_extractor.v2(
    input_data=m4.data,
    features=m3.data,
    date_col='date',
    instrument_col='instrument'
)

m7 = M.join.v3(
    data1=m2.data,
    data2=m5.data,
    on='date,instrument',
    how='inner',
    sort=False
)

m13 = M.dropnan.v1(
    input_data=m7.data
)

m6 = M.stock_ranker_train.v5(
    training_ds=m13.data,
    features=m3.data,
    learning_algorithm='排序',
    number_of_leaves=30,
    minimum_docs_per_leaf=1000,
    number_of_trees=20,
    learning_rate=0.1,
    max_bins=1023,
    feature_fraction=1,
    m_lazy_run=False
)

m9 = M.instruments.v2(
    start_date=T.live_run_param('trading_date', '2015-01-01'),
    end_date=T.live_run_param('trading_date', '2017-11-15'),
    market='CN_STOCK_A',
    instrument_list='',
    max_count=0
)

m10 = M.general_feature_extractor.v6(
    instruments=m9.data,
    features=m3.data,
    start_date='',
    end_date='',
    before_start_days=0
)

m11 = M.derived_feature_extractor.v2(
    input_data=m10.data,
    features=m3.data,
    date_col='date',
    instrument_col='instrument'
)

m14 = M.dropnan.v1(
    input_data=m11.data
)

m8 = M.stock_ranker_predict.v5(
    model=m6.model,
    data=m14.data,
    m_lazy_run=False
)

# 回测引擎：每日数据处理函数，每天执行一次
def m12_handle_data_bigquant_run(context, data):
    # 按日期过滤得到今日的预测数据
    ranker_prediction = context.ranker_prediction[
        context.ranker_prediction.date == data.current_dt.strftime('%Y-%m-%d')]

    # 1. 资金分配
    # 平均持仓时间是hold_days，每日都将买入股票，每日预期使用 1/hold_days 的资金
    # 实际操作中，会存在一定的买入误差，所以在前hold_days天，等量使用资金；之后，尽量使用剩余资金（这里设置最多用等量的1.5倍）
    is_staging = context.trading_day_index < context.options['hold_days'] # 是否在建仓期间（前 hold_days 天）
    cash_avg = context.portfolio.portfolio_value / context.options['hold_days']
    cash_for_buy = min(context.portfolio.cash, (1 if is_staging else 1.5) * cash_avg)
    cash_for_sell = cash_avg - (context.portfolio.cash - cash_for_buy)
    positions = {e.symbol: p.amount * p.last_sale_price
                 for e, p in context.perf_tracker.position_tracker.positions.items()}

    # 2. 生成卖出订单：hold_days天之后才开始卖出；对持仓的股票，按StockRanker预测的排序末位淘汰
    if not is_staging and cash_for_sell > 0:
        today = data.current_dt
        today_str=str(today.date())
        # 持仓股票列表，为字符串
        equities = {e.symbol: p for e, p in context.portfolio.positions.items() if p.amount>0}
         # 调仓：卖出所有持有股票
        for instrument in equities:
            # 停牌的股票，将不能卖出，将在下一个调仓期处理
            if today-equities[instrument].last_sale_date>=datetime.timedelta(context.options['hold_days']-2) and data.can_trade(context.symbol(instrument)):
                 context.order_target_percent(context.symbol(instrument), 0)
            
     

    # 3. 生成买入订单：按StockRanker预测的排序，买入前面的stock_count只股票
    buy_cash_weights = context.stock_weights
    buy_instruments = list(ranker_prediction.instrument[:len(buy_cash_weights)])
    max_cash_per_instrument = context.portfolio.portfolio_value * context.max_cash_per_instrument
    for i, instrument in enumerate(buy_instruments):
        cash = cash_for_buy * buy_cash_weights[i]
        if cash > max_cash_per_instrument - positions.get(instrument, 0):
            # 确保股票持仓量不会超过每次股票最大的占用资金量
            cash = max_cash_per_instrument - positions.get(instrument, 0)
        if cash > 0:
            context.order_value(context.symbol(instrument), cash)

# 回测引擎：准备数据，只执行一次
def m12_prepare_bigquant_run(context):
    pass

# 回测引擎：初始化函数，只执行一次
def m12_initialize_bigquant_run(context):
    # 加载预测数据
    context.ranker_prediction = context.options['data'].read_df()

    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    # 预测数据，通过options传入进来，使用 read_df 函数，加载到内存 (DataFrame)
    # 设置买入的股票数量，这里买入预测股票列表排名靠前的5只
    stock_count = 5
    # 每只的股票的权重，如下的权重分配会使得靠前的股票分配多一点的资金，[0.339160, 0.213986, 0.169580, ..]
    context.stock_weights = T.norm([1 / math.log(i + 2) for i in range(0, stock_count)])
    # 设置每只股票占用的最大资金比例
    context.max_cash_per_instrument = 0.2
    context.options['hold_days'] = 2

m12 = M.trade.v3(
    instruments=m9.data,
    options_data=m8.predictions,
    start_date='',
    end_date='',
    handle_data=m12_handle_data_bigquant_run,
    prepare=m12_prepare_bigquant_run,
    initialize=m12_initialize_bigquant_run,
    volume_limit=0,
    order_price_field_buy='open',
    order_price_field_sell='close',
    capital_base=1000000,
    benchmark='000300.SHA',
    auto_cancel_non_tradable_orders=True,
    data_frequency='daily',
    price_type='后复权',
    plot_charts=True,
    backtest_only=False
)
