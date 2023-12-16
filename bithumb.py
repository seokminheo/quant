current_price = pybithumb.get_current_price(str(symbol)[:-4])
unit = bithumb.get_balance(str(symbol)[:-4])[0]

exchange.create_market_sell_order(
        symbol=symbol,
        amount=math.trunc(unit*10000)/10000
        )
amount = math.trunc(unit*10000)/10000*current_price
