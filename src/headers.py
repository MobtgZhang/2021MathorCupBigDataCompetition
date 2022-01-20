def get_headers_quesA():
    header_list = [
        "carid",
        "tradeTime",
        "brand",
        "serial",
        "model",
        "mileage",
        "color",
        "cityId",
        "carCode",
        "transferCount",
        "seatings",
        "registerDate",
        "licenseDate",
        "country",
        "maketype",
        "modelyear",
        "displacement",
        "gearbox",
        "oiltype",
        "newprice",
        "anonymousFeature",
        "price"
    ]
    meaning_list = [
        "车辆id",
        "展销时间",
        "品牌id",
        "车系id",
        "车型id",
        "里程",
        "车辆颜色",
        "车辆所在城市id",
        "国标码",
        "过户次数",
        "载客人数",
        "注册日期",
        "上牌日期",
        "国别",
        "厂商类型",
        "年款",
        "排量",
        "变速箱",
        "燃油类型",
        "新车价",
        "15个匿名特征",
        "价格"
    ]
    tp_list = []
    for k in range(15):
        tp_list.append("anonymousFeature%d"%(k+1))
    tp_name_list = []
    for k in range(15):
        tp_name_list.append("匿名特征%d"%(k+1))
    header_list = header_list[0:20] + tp_list + [header_list[-1]]
    meaning_list = meaning_list[0:20] + tp_name_list + [meaning_list[-1]]
    return header_list,meaning_list
def get_headers_quesB():
    name_list = [
        "carid",
        "pushDate",
        "pushPrice",
        "updatePriceTimeJson",
        "pullDate",
        "withdrawDate "
    ]
    meaning_list = [
        "车辆id",
        "上架时间",
        "上架价格",
        "{价格调整时间：调整后价格}",
        "下架时间(成交车辆下架时间和成交时间相同)",
        "成交时间"
    ]
    return name_list,meaning_list
