import json

def transform_json(input_file, output_file):
    """
    将原始JSON格式转换为目标格式
    """
    try:
        # 读取原始JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换数据
        transformed_data = []
        
        for idx, item in enumerate(data):
            # 构建新的数据结构，保持原始question内容
            new_item = {
                "question_id": idx,
                "image": item.get('image', ''),
                "text": item.get('text', ''),
                "category": "default"
            }
            
            transformed_data.append(new_item)
        
        # 写入转换后的数据，每行一个JSON对象
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in transformed_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"转换完成！共处理 {len(transformed_data)} 条数据")
        print(f"输出文件：{output_file}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except json.JSONDecodeError:
        print(f"错误：{input_file} 不是有效的JSON文件")
    except Exception as e:
        print(f"错误：{e}")

def transform_json_from_string(json_string):
    """
    直接从JSON字符串转换（用于测试）
    """
    try:
        data = json.loads(json_string)
        
        transformed_data = []
        
        for idx, item in enumerate(data):
            new_item = {
                "question_id": idx,
                "image": item.get('image', ''),
                "text": item.get('question', ''),
                "category": "default"
            }
            
            transformed_data.append(new_item)
        
        # 输出每行一个JSON对象的格式
        for item in transformed_data:
            print(json.dumps(item, ensure_ascii=False))
            
    except json.JSONDecodeError:
        print("错误：输入不是有效的JSON格式")
    except Exception as e:
        print(f"错误：{e}")

# 使用示例
if __name__ == "__main__":
    # 方法1：从文件转换
    transform_json('/home/users/nus/ophv119/LLaVA/playground/data/eval/df_db/testset/llava_test.jsonl', '/home/users/nus/ophv119/LLaVA/playground/data/eval/df_db/testset/llava_test_1.jsonl')
