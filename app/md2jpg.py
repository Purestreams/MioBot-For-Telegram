import markdown2
from playwright.async_api import async_playwright
import datetime
import os
from PIL import Image
import asyncio

async def md_to_image(md_text, theme='cute_anime', output_path='output.png', width=550):
    """
    Converts a Markdown string to an image with a specified theme.

    :param md_text: The Markdown text to convert.
    :param theme: The theme to use ('cute_anime' or 'formal_code').
    :param output_path: The path to save the output image.
    :param width: The width of the content area in pixels.
    """
    # Get current date
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    # Convert Markdown to HTML
    html_body = markdown2.markdown(md_text, extras=["fenced-code-blocks", "tables"])

    # CSS for themes
    base_css = f"""
    @import url('https://fonts.googleapis.com/css2?family=Noto+Emoji:wght@300..700&family=Noto+Sans+SC:wght@100..900&family=Noto+Sans+TC:wght@100..900&family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap');
    body {{ font-family: "Noto Sans TC", "Noto Sans SC", "Noto Emoji", sans-serif; padding: 40px; background-color: #f9f9f9; }}
    .container {{ max-width: {width}px; margin: 0 auto; background-color: white; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 30px; }}
    .footer {{ text-align: right; font-size: 12px; color: #888; margin-top: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background-color: #f2f2f2; }}
    pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; }}
    code {{ font-family: 'Courier New', Courier, monospace; }}
    blockquote {{ border-left: 4px solid #ccc; padding-left: 15px; color: #666; }}
    """

    cute_anime_css = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Emoji:wght@300..700&family=Noto+Sans+SC:wght@100..900&family=Noto+Sans+TC:wght@100..900&family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap');
    body { background-color: #ffefff; font-family: "Noto Sans TC", "Noto Sans SC", "Noto Emoji", sans-serif; }
    .container { background-color: #ffffff; border: 2px dashed #ffc0cb; }
    h1, h2, h3 { color: #e85c90; }
    pre { background-color: #fff0f5; border: 1px solid #ffc0cb; }
    """

    formal_code_css = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Emoji:wght@300..700&family=Noto+Sans+SC:wght@100..900&family=Noto+Sans+TC:wght@100..900&family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap');
    body { font-family: "Noto Sans TC", "Noto Sans SC", "Noto Emoji", sans-serif; background-color: #f0f2f5; }
    .container { border: 1px solid #e0e0e0; }
    h1, h2, h3 { color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px;}
    code, pre { font-family: 'Fira Code', monospace, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"; }
    pre { background-color: #2d2d2d; color: #f8f8f2; border-radius: 5px; }
    /* Basic syntax highlighting for demo */
    .code-keyword { color: #ff79c6; }
    .code-string { color: #f1fa8c; }
    .code-comment { color: #6272a4; }
    .code-function { color: #50fa7b; }
    """

    theme_css = cute_anime_css if theme == 'cute_anime' else formal_code_css

    # Full HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            {base_css}
            {theme_css}
        </style>
    </head>
    <body>
        <div class="container">
            {html_body}
            <div class="footer">
                Telegram: @MioooooooooBot, Made by Mio &bull; {date_str}
            </div>
        </div>
    </body>
    </html>
    """

    # Use Playwright to render HTML and take a screenshot
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(device_scale_factor=4)
        await page.set_content(html_content)
        
        # Set a viewport width. Height will be determined by full_page screenshot.
        viewport_width = width + 80  # width + padding
        await page.set_viewport_size({ "width": viewport_width, "height": 100 }) # Initial height, will be ignored by full_page

        # Screenshot is always PNG, create a temporary path for it
        temp_png_path = output_path + ".png"
        await page.screenshot(path=temp_png_path, full_page=True)
        await browser.close()

    # Open the PNG and save as JPG
    img = Image.open(temp_png_path)
    # Ensure image is in RGB mode for saving as JPEG
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    # Determine final output path, assuming JPG if no extension
    final_output_path = output_path
    if not final_output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif')):
        final_output_path += '.jpg'

    if final_output_path.lower().endswith(('.jpg', '.jpeg')):
         img.save(final_output_path, 'JPEG', quality=40, optimize=True, subsampling=2)
         print(f"Image saved and compressed to {final_output_path}")
    elif final_output_path.lower().endswith('.webp'):
        img.save(final_output_path, 'webp', quality=40, optimize=True)
        print(f"Image saved and compressed to {final_output_path}")
    elif final_output_path.lower().endswith('.avif'):
        img.save(final_output_path, 'AVIF', quality=40, optimize=True)
        print(f"Image saved and compressed to {final_output_path}")
    else: # It's a png
        # In case the original output_path was already a .png, just rename temp file
        os.rename(temp_png_path, final_output_path)
        print(f"Image saved to {final_output_path}")

    # Clean up the temporary PNG file if it still exists and is different from final path
    if os.path.exists(temp_png_path) and temp_png_path != final_output_path:
        os.remove(temp_png_path)


if __name__ == '__main__':
    # Example Usage
    markdown_example = """
# My Awesome Document

This is a test document to showcase the Markdown to image conversion.

Artificial Intelligence (AI) is revolutionizing the modern world. It powers advancements in fields like healthcare, transportation, and finance by enabling machines to analyze data and make decisions.

Key aspects of AI include:
- Machine Learning: Systems that learn from data.
- Natural Language Processing: Understanding and generating human language.
- Computer Vision: Interpreting and processing visual information.

As AI continues to evolve, it presents both opportunities and challenges, calling for innovative solutions and responsible practices.

## Features
- Supports lists
- And tables
- And code blocks!

### A Cute Table
| Character | Series         |
|-----------|----------------|
| Anya      | Spy x Family   |
| Kaguya    | Kaguya-sama    |



### Code Block
```python
def hello_world():
    print("Hello, from Mio!")
```

> A famous quote someone once said.
"""


    markdown_example_2 = """
# 如何描述中科大具有中科大特有科气的男生「科男」的特点？

作者：ShirokumaQWQ  
链接：https://www.zhihu.com/question/553340006/answer/3322736707  
来源：知乎  
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

我终于差不多搞明白了这件事的原因：科大的具有强抽象思维的阿斯伯格综合征比例显著高于一般人群的均值，特别是数物少计（以上排名分先后）。也就是说：所谓的"科气"并不是由于科大这个地方风水导致的，可能是惯用抽象思维的阿斯伯格的外在体现，以及当环境中此类人比例过高时，其他人外在被同化的现象。下面我以 "假设&定义->推论&例子”的方式来说明这个观点。由于目前的研究并不能严格指出阿斯伯格是大脑的哪一部分异于常人发育或其他原因而造成的，所以我们暂且把它定义为满足出现一类的特征的人的集合。由于对于阿斯伯格的评审有不同的标准，我们尽量取不同标准的交集用于定义，并把一些被发现强相关性但并不作为判别标准的特征放在假设中。

**Definition 1 (阿斯伯格 / Asperger)**  
我们把具有以下具有特称的人群称为阿斯伯格综合征，记作AS：社交互动与沟通异常。难以理解非语言交流（例如表情与肢体动作）表达的内容，语言交流中只能判断字面意思。*重复的刻板行为。在固定计划被打乱时比常人更暴躁易怒。与其他孤独症谱系不同，AS有社交意愿。口语语言发育正常，智力发育无异常。*一部分AS会过分高估自己理解话外音能力，例如：因为自己马上就能领会到苏联笑话，而自认为这种能力没有问题，事实上这完全不是一个意思。我将使用一个例子来说明什么叫做“能理解他人的话外音”，见附录。

**Definition 2 (神经典型者 / Neurotypical)**  
泛指无神经学特异表现的人：换言之，即无自闭症、 阅读障碍 、 发展性协调障碍 、 双相情感障碍 、 注意力缺陷过动症 ，或其他类似情况的人，记作NT。AS通常执着于一个或数个领域。对于狭隘主题的强烈关注、韵律受限和身体笨拙是该病的典型特征，但不是诊断的必要条件。

所以我们作出以下假设：

**Assumption 3**  
对逻辑推导有较高需求的领域（例如数学、理论物理、计算机等）出现AS的概率显著高于在一般人群中的平均概率。有调查发现，在TODO

P.S.（非常重要）：在补充这部分的依据的时候，我突然意识到我可能搞错了一件事：我可能搞错了因果关系——应该是完全发展的抽象思维能力导致了无法进行通常的有效社交，也就是assumption 3可以推导出定义1.1，而不是反过来代表一些相关性。 这种抽象思维能力（有点类似荣格中的Ti），我倾向于是大脑在建立不同事物的连接时，建立的依据是因果关系——类似有向图的模式，而具象的思维则更接近于考察不同事件的相关性，像是无向图。而这种思维方式会直接导致对使用具象思维人的语言的错误分析，即定义1.1。也许需要修改定义方式，自闭症/阿斯伯格只是对现象的总结而非对原因的分析，无论是抽象思维规则过度发展还是具象思维，都可能导致人际交往困难等现象，但具体的诱因是完全不同的。（TODO。。有时间再调研分析

所以这里让后文的AS是代指具有强抽象思维导致社交问题的人群。

**Assumption 4**  
AS与身体协调性较差具有正相关性。TODO：reference

e.g. 4.1 科带的体育成绩显著差于其他高校。这通常被认为是大学生缺乏锻炼，但一个值得注意的地方是：是因为不擅长运动才不运动，还是因为不运动才不擅长运动？我个人更倾向于前者。自认为体育较差的同学可以回忆下，自己是因为长时间不运动才这样吗？是否从小在与同龄人玩耍的时候就已经表现出了一定程度的运动笨拙？缺乏锻炼常常体现在一些跟心肺功能/力量的运动表现（例如跑步等）较差上，而身体不协调则会体现在一些技巧性的运动上：例如跳远、球类运动的不擅长。

e.g 4.2 糟糕的字迹。带过一次数院课程的助教，批改作业让我十分痛苦，90%的作业本字迹属于正常人群中后10%的水平。这可能与拙劣的身体协调性相关。关于这点同样一个常见的谬误是：认为文科卷面重要、会直接影响得分，而理科不重要、不影响得分，所以这些成绩好的理科生卷面更糟糕。这是个推导试图否认“擅长数理”与“卷面更差”的相关性。但即便不考虑卷面过差可能带来的误判以及文理科都要考语文：我们假设理科生成绩跟卷面无关，那么写字水平不同的人应当是均匀分布在不同分数段的，至少决不应该出现成绩更好而卷面更差的现象。但事实会更有趣一点——并不是简单的理科更好则卷面更差。只是为了应付高考的话，事实上不需要过强的逻辑推理能力。比较数院的录取分数线附近的所有考生，数学专业的同学的字迹显著地更为糟糕。这里的结论是：更加擅长或是说热衷于逻辑思辨 与 糟糕的身体协调性 出现了有趣的相关性。（这个结论是有统计支持的，不是我编的，但是原因？

**Theorem 5.**  
AS在人际交往中总是滔滔不绝地谈论自己的兴趣，但并不能意识到别人是否对这个话题感兴趣。  
Proof：来自对他人语言理解异常与行为的错误判断，此外由于自己在某些方面（抽象思维）非常擅长，总能自行得出许多非平凡的结论，加之又有社交需求，导致总是希望对人输出自己的分析。另一方面可能也来自希望他人能给自己同样激烈的反馈来修正自己的逻辑体系。

e.g. 5.1喜欢跟逢人就吹数学物理键政。  
e.g. 5.2 后来发现尬吹数学物理并不招人待见，开始跟人尬聊“天空为什么是蓝色的”，并认为谈论这个话题的自己如同诗人一样浪漫。（P.S. 不是我说尬啊，上面高赞说的

**Proposition 6 糟糕的外观。**  
Proof：糟糕的外观来自于  
1. 薄弱的具象思考能力，不易把外观跟人的其他性质联系起来。具象思维的人更倾向于建立这种相关性联系，认为衣着、体态等可能与对象的其他品质有关，从而得出第一印象。但强抽象思维的人，不认为外观与对象的其他品质之间有因果关系，不会这么评判他人，同时也不认为自己这样。（常见于努力否定穿着打扮的必要性。  
2. 对他人反馈感受弱。由于def 1.1，难以感受或者理解到他人对自己外观的负反馈，从而不会改变自己。

**Proposition 7 不喜线下社交，但在线上社交表现活跃。**  
Pf：答案到这里基本已经可以说显然了。如果一个人有社交需求，但无法很好地理解他人的表情与话外音 def 1.1，同时热衷于发表自己的观点 theorem 5，并且伴有外表糟糕 proposition 6、不擅长体育运动 assumption 4的特征，那么毫无疑问，线上社交的舒适程度会远高于线下社交。线下社交容易引起人（具象思维人）对外表的消极评价，同时不擅长体育运动导致厌恶以参加户外活动，且在与人沟通的时候总是不断地打断别人，发表自己观点，导致很难受到人群的欢迎。而线上社交并不需要评判外表（人人都是美少女头像），也避免了体育运动。此外线上社交具备无序性——不像是线下社交必须每个人依次说话，无论是群聊还是论坛，都可以不断地发表自己的观点而不在乎他人是否感兴趣/被打断。

**Corollary 7.1 宅，喜欢二次元。**  
二次元本身就是抽象思维的一种应用，将人物的美好特征集中抽象于角色上。（我瞎扯的，我更倾向于是proposition 7：二次元不需要线下社交，但有很大的线上社交空间。

e.g.7.1.1 妮可galgame交流群源自于数院群，且群内大量充斥抽象、鉴证与交流数学题目，属于科气结晶典型现象之一。

**Theorem 8**  
随着年龄的提升，AS会逐渐学习以掩饰自己的不擅长社交。说实话这件事是困难的，很多该听不懂的话（见附录）其实还是听不懂。但是通过不断地学习、反馈，还是能强行得到并记住一些相关性。例如：如何识别人是不是在说反话。

**Proposition 9 本来不怎么"科气"的同学来了科大之后变得更"科"了**  
一种情况我倾向是本身即是AS，但由于theorem 8，在高中可能有一些同学掩饰了自己的显著特征，在科大对”情商“要求较为宽松的环境中又回归了本质。另一种情况可能是在发现科大没有人在意外在的条件下，缺少正反馈，也放弃个人外观修饰了，但这种情况不会影响本身情感识别能力的特征，只是属于外观上的表现。

**Proposition 10 "科气"不止在科大出现，在北大、复旦等学校的数院也出现了类似的现象。**  
TODO：reference，某答主回答。

根据以上论述，我们认为"科气"是来自于指的是惯用抽象思维的阿斯伯格的外在体现，这证明了科气不是科大学校本身的区域buff，而是因为筛选了过多强抽象思维导致社交障碍的同学的结果，所以这当然不会只在科大出现，类似的筛选方式都会导致相同的结果。

---

### 附录

这里给出两个出现无法正确理解他人语言表达的典型案例：

E.g. 1（背景：”我“是一名小学生，且课桌常年非常乱  
某天上课，班主任讲到课桌整洁问题，又把我当作反面典型进行批评。我十分害怕，赶紧开始收拾课桌。此时，班主任更加生气：”你现在想起来收拾了？你要收给我滚到外面收去“。我不明白她为什么要我出去收拾，我推测可能是怕吵到班里同学，或是像走廊罚站一样惩罚我去外面收拾。但我不敢问，更不敢违抗，只好老老实实地把课桌往教室外面拖。此时班主任更加生气，一脚把我的课桌踹翻教师在外面，让我收完了才能进去。  
问：为什么班主任这么生气？

E.g. 2 （背景：”我“是一名语文成绩比其他学科成绩更糟糕的初中生。我在语文考试中因为答错了很多老师认为的简单题而得到了糟糕的成绩，此时在老师办公室：老师指向卷面上的某道题目问我：”你怎么这个题也在错。一个优秀的学生是不该错这种题目的对不对？“我：”对。“老师：“那你为什么错？”我。。。。。难道是因为我不优秀吗，逆否命题就是“错了这种题目的学生不优秀”吧，但我不愿意这么说，我认为亲口承认我不优秀有辱我的尊严。老师看我没回应，又问了一句：“你为什么错？？”我：“因为我不优秀”老师：？？？  
问：老师想让我说什么？

"""

async def main_async():
    # Example usage of the md_to_image function
    #await md_to_image(markdown_example, theme='cute_anime', output_path='output_cute_anime.jpg')
    #await md_to_image(markdown_example_2, theme='formal_code', output_path='output/output_formal_code.webp')
    await md_to_image(markdown_example_2, theme='formal_code', output_path='output/output_formal_code.jpg')
    await md_to_image(markdown_example_2, theme='formal_code', output_path='output/output_formal_code.AVIF')

if __name__ == '__main__':
    asyncio.run(main_async())