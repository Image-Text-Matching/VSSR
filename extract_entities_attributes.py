import json

def extract_entities_attributes(json_file, output_file):
    # Load JSON data
    with open(json_file) as f:
        data = json.load(f)

    results = []

    # Iterate through each scene graph
    for scene in data:
        scene_result = {
            'caption': scene['caption'],
            'entities': []
        }

        # Extract entities and attributes
        entities = scene['entities']
        for entity in entities:
            name = entity['name']
            attributes = entity['attributes']
            scene_result['entities'].append({
                'name': name,
                'attributes': attributes
            })

        results.append(scene_result)

    # 保存结果到新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results have been saved to {output_file}")
    return results


# Example usage:
# 处理第一个文件
results_image_caption = extract_entities_attributes(
    'D:\\study\\FACTUAL\\scene_graph\\test_caps_synthesis_florence_det.json',
    'D:\\study\\FACTUAL\\scene_graph\\test_caps_synthesis_florence_det_extract_ea.json'
)

# 处理第二个文件
results_text = extract_entities_attributes(
    'D:\\study\\FACTUAL\\scene_graph\\test_caps.json',
    'D:\\study\\FACTUAL\\scene_graph\\test_caps_extract_ea.json'
)