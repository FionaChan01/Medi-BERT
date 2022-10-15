# from typing import Optional
#
# from fastapi import FastAPI
# from pydantic.main import BaseModel
#
# app = FastAPI()
#
#
# def EXEC(str):  # 这里是你要调用的模型，str是传进来的病人口述。
#     print(str)
#     uft_str = str.encode("iso-8859-1").decode('gbk').encode('utf8')
#     print("utf_str",uft_str)
#     if (str):
#         return "hello_world," + str
#     else: return "hello_world," + "fail!"
#
# class ansTest:BaseModel
#
#
# @app.get("/find/")
# async def handleWords(words: Optional[str] = None):
#     return {"str": EXEC(words)}
#
# @app.post("/ans/")
# async def handle()
#
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from transformers import BertConfig, TextClassificationPipeline,BertTokenizer, BertModel, BertAdapterModel
import torch



config = BertConfig.from_pretrained(
    "C:\\Users\\Plasw\\Desktop\\model\\mcbert",
    num_labels=36,
)
model = BertAdapterModel.from_pretrained(
    "C:\\Users\\Plasw\\Desktop\\model\\mcbert",
    config=config,
)

tokenizer = BertTokenizer.from_pretrained("C:\\Users\\Plasw\\Desktop\\model\\mcbert")
model.load_state_dict(torch.load('C:\\Users\\Plasw\\Desktop\\model\\bert.ckpt'),False)




model.load_adapter('C:\\Users\\Plasw\\Desktop\\model\\checkpoint-5500\\symptom_adapter',set_active = True)
model.eval()
#model.add_classification_head(
#    'C:\\Users\\Plasw\\Desktop\\model\\checkpoint-7000\\symptom_adapter',
#    num_labels=36,
#    id2label= {19:"消化内科",16:"整形美容科",27:"耳鼻喉科",14:"心胸外科",18:"泌尿外科",30:"肾内科",21:"男科",3:"产科",23:"眼科",35:"骨外科",5:"儿科综合",28:"肛肠科",29:"肝胆外科",13:"心理科",34:"风湿免疫科",15:"性病科",24:"神经内科",31:"肿瘤科",33:"遗传病科",8:"呼吸内科",9:"妇科",26:"精神科",20:"烧伤科",6:"内分泌科",22:"皮肤科",17:"普外科",7:"口腔科",0:"不孕不育",12:"心内科",2:"中医骨伤科",32:"血液科",1:"中医综合",4:"传染科",11:"小儿外科",25:"神经外科",10:"小儿内科"},
#    overwrite_ok = True
#  )
#model.set_active_adapters("symptom_adapter")
#model.train_adapter("symptom_adapter")



classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许访问的源
    allow_credentials=True,  # 支持 cookie
    allow_methods=["*"],  # 允许使用的请求方法
    allow_headers=["*"]  # 允许携带的 Headers
)
router = APIRouter()


def EXEC(str):  # 这里是你要调用的模型，str是传进来的病人口述。
    print(str)
    # uft_str = str.encode("iso-8859-1").decode('gbk').encode('utf8')
    # print("utf_str", uft_str)
    if (str):
        return classifier(str)[0]['label']
    else:
        return "fail!"


class Voice(BaseModel):
    """
    id为可选项
    """
    id: Optional[int] = 0
    msg: str = ""


class Res(BaseModel):
    code: int = 200
    msg: str = ""
    data: dict = {}


@router.post("/getDept")
async def getdept(v: Voice) -> Res:
    # 获取语音文本
    con = v.msg
    # 返回结果
    res = Res()
    try:
        # 在此填写业务逻辑
        pass
    except Exception:
        res.code = 400
        res.msg = "失败"
    else:
        res.code = 200
        res.msg = EXEC(con)
    return res


if __name__ == "__main__":
    import uvicorn

    app.include_router(router,prefix="/triage")
    uvicorn.run(app, host="127.0.0.1", port=10000)
