#!/usr/bin/env python3
"""
General Object Detector (Batch Mode)
This script analyzes images in batch mode and detects specific object types.

It randomly selects images from the specified directory, detects objects with locations,
and generates image-text corresponding reports for each image.
"""

import os
import argparse
import base64
import requests
import random
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import configuration
from config import (
    API_KEY, BASE_URL, MODEL_NAME, MAX_TOKENS,
    SUPPORTED_OBJECT_TYPES, SUPPORTED_IMAGE_FORMATS,
    DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, KNOWLEDGE_BASE_PATH,
    RAG_TOP_K, MAX_CHUNK_SIZE, CHUNK_OVERLAP, PROMPT_TEMPLATE,
    object_type_abbreviations
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Detect specific object types in batch mode")
    # 批处理模式参数
    parser.add_argument("--dir", type=str, default=DEFAULT_INPUT_DIR,
                        help="Directory containing images to analyze")
    parser.add_argument("--file", type=str, default="",
                        help="Specific image file to analyze (overrides --dir and --count)")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of random images to select from the directory (use 0 to process all images)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for the detection reports")
    parser.add_argument("--format", type=str, choices=["md", "txt"], default="md",
                        help="Output format for the detection reports (markdown or text)")
    parser.add_argument("--api-key", type=str, default=API_KEY,
                        help="SiliconFlow API key")
    parser.add_argument("--base-url", type=str, default=BASE_URL,
                        help="SiliconFlow API base URL")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip processing if report already exists in output directory")
    return parser.parse_args()

def get_image_files(directory):
    """Get all supported image files from the directory"""
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_IMAGE_FORMATS:
                image_files.append(os.path.join(root, file))
    
    return image_files

def random_select_images(image_files, count):
    """Randomly select a specified number of images"""
    if len(image_files) <= count:
        print(f"Note: Only {len(image_files)} images found, using all of them")
        return image_files
    
    selected = random.sample(image_files, count)
    print(f"Randomly selected {len(selected)} images from {len(image_files)} total images")
    return selected

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def semantic_chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks based on semantic boundaries (sentences and paragraphs)"""
    chunks = []
    current_chunk = ""
    
    # Split text into paragraphs first
    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If paragraph is short enough, add it directly
        if len(current_chunk) + len(paragraph) + 1 <= max_chunk_size:
            if current_chunk:
                current_chunk += '\n'
            current_chunk += paragraph
        else:
            # Paragraph is too long, split by sentences
            # Sentence delimiters
            sentence_delimiters = ['. ', '! ', '? ', '。', '！', '？']
            
            # Split paragraph into sentences
            sentences = [paragraph]
            for delimiter in sentence_delimiters:
                new_sentences = []
                for sentence in sentences:
                    parts = sentence.split(delimiter)
                    for i, part in enumerate(parts):
                        if i < len(parts) - 1:
                            new_sentences.append(part + delimiter)
                        else:
                            new_sentences.append(part)
                sentences = new_sentences
            
            # Process sentences
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                    if current_chunk:
                        current_chunk += ' '
                    current_chunk += sentence
                else:
                    # Current chunk is full, add to chunks
                    if current_chunk:
                        chunks.append(current_chunk)
                        # Start new chunk with overlap
                        current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                    
                    # If sentence is still too long, split by words
                    if len(sentence) > max_chunk_size:
                        # Split by spaces
                        words = sentence.split(' ')
                        temp_chunk = ""
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                                if temp_chunk:
                                    temp_chunk += ' '
                                temp_chunk += word
                            else:
                                chunks.append(temp_chunk)
                                temp_chunk = word
                        if temp_chunk:
                            current_chunk = temp_chunk
                    else:
                        current_chunk = sentence
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def create_embeddings(texts):
    """Create embeddings for text chunks using TF-IDF"""
    vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2))
    embeddings = vectorizer.fit_transform(texts)
    return embeddings, vectorizer

def retrieve_relevant_chunks(query, chunks, embeddings, vectorizer, top_k=3):
    """Retrieve top-k relevant chunks based on cosine similarity"""
    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return relevant chunks with their scores
    relevant_chunks = [(chunks[i], similarities[i]) for i in top_indices]
    return relevant_chunks

def rerank_chunks(chunks_with_scores, query, damage_types):
    """Rerank chunks based on multiple factors"""
    reranked = []
    
    for chunk, score in chunks_with_scores:
        # Factor 1: Original similarity score
        factor1 = score
        
        # Factor 2: Presence of damage type keywords
        factor2 = 0
        for damage_type in damage_types:
            if damage_type in chunk:
                factor2 += 0.2
        
        # Factor 3: Length of chunk (prefer medium length chunks)
        chunk_length = len(chunk)
        if 200 <= chunk_length <= 400:
            factor3 = 0.1
        elif 100 <= chunk_length < 200 or 400 < chunk_length <= 500:
            factor3 = 0.05
        else:
            factor3 = 0
        
        # Factor 4: Query term density
        query_terms = query.split()
        term_count = sum(1 for term in query_terms if term in chunk)
        factor4 = term_count / len(query_terms) if query_terms else 0
        
        # Calculate final score
        final_score = factor1 + factor2 + factor3 + factor4
        reranked.append((chunk, final_score))
    
    # Sort by final score
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked

def load_knowledge_base(knowledge_path=KNOWLEDGE_BASE_PATH):
    """Load knowledge base from file"""
    try:
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            knowledge_content = f.read()
        print(f"Loaded knowledge base from: {knowledge_path}")
        print(f"Knowledge base size: {len(knowledge_content)} characters")
        return knowledge_content
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return ""

def get_relevant_knowledge(knowledge_content, query, object_types, top_k=RAG_TOP_K):
    """Get relevant knowledge using RAG with reranking"""
    if not knowledge_content:
        return ""
    
    # Split knowledge into semantic chunks
    chunks = semantic_chunk_text(knowledge_content)
    print(f"Split knowledge base into {len(chunks)} semantic chunks")
    
    if not chunks:
        return ""
    
    # Create embeddings
    embeddings, vectorizer = create_embeddings(chunks)
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, chunks, embeddings, vectorizer, top_k=top_k)
    print(f"Retrieved {len(relevant_chunks)} relevant chunks")
    
    # Rerank chunks
    reranked_chunks = rerank_chunks(relevant_chunks, query, object_types)
    print(f"Reranked chunks with scores: {[(i+1, score) for i, (_, score) in enumerate(reranked_chunks)]}")
    
    # Combine top chunks
    combined_knowledge = "\n".join([chunk for chunk, _ in reranked_chunks])
    print(f"Combined knowledge size: {len(combined_knowledge)} characters")
    
    return combined_knowledge

def analyze_image_with_siliconflow(image_path, api_key, base_url):
    """Analyze image using SiliconFlow API with format conversion"""
    try:
        # Store original image path
        original_image_path = image_path
        
        # Check if image is MPO format
        if image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
            # Try to open with PIL to check actual format
            with Image.open(image_path) as img:
                if img.format == 'MPO':
                    # Convert MPO to JPEG
                    print("Converting MPO format to JPEG...")
                    # Create temporary JPEG file
                    temp_jpeg = image_path.rsplit('.', 1)[0] + '_temp.jpg'
                    img.save(temp_jpeg, 'JPEG')
                    image_path = temp_jpeg
        
        base64_image = encode_image(image_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Load knowledge base
        knowledge_content = load_knowledge_base()
        
        # Get actual image dimensions
        with Image.open(image_path) as img:
            actual_width, actual_height = img.size
        
        # Create query for RAG
        rag_query = f"目标检测：识别并定位以下目标类型：{', '.join(SUPPORTED_OBJECT_TYPES)}"
        
        # Get relevant knowledge with reranking
        relevant_knowledge = get_relevant_knowledge(knowledge_content, rag_query, SUPPORTED_OBJECT_TYPES)
        
        # Create prompt with knowledge base
        if relevant_knowledge:
            knowledge_reference = f"**相关知识库参考：**\n{relevant_knowledge}"
        else:
            knowledge_reference = ""
        
        prompt = PROMPT_TEMPLATE.format(
            object_types=', '.join(SUPPORTED_OBJECT_TYPES),
            width=actual_width,
            height=actual_height,
            knowledge_reference=knowledge_reference
        )
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": MAX_TOKENS
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            # Return both detection result and the image path used for analysis
            return data['choices'][0]['message']['content'], image_path
        else:
            print(f"Image Analysis API Error: {response.status_code} - {response.text}")
            return f"Error analyzing image: {response.status_code}", image_path
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return f"Error analyzing image: {str(e)}", image_path

def draw_object_boxes(image_path, detection_result, output_dir):
    """Draw bounding boxes around detected objects and save the processed image"""
    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")
        
        # Import PIL modules for drawing
        from PIL import ImageDraw, ImageFont
        
        # Create draw object
        draw = ImageDraw.Draw(image)
        
        # Extract object types and locations from detection result
        object_info = extract_object_info(detection_result)
        
        # Get image dimensions
        width, height = image.size
        print(f"Debug: Image dimensions: width={width}, height={height}")
        
        # Process each detected object
        print(f"Debug: Processing {len(object_info)} objects")
        for i, obj in enumerate(object_info, 1):
            object_type = obj.get('type', f"Object {i}")
            location = obj.get('location', "")
            bbox_coords = obj.get('bbox', None)
            
            # Get abbreviation for object type
            label = object_type_abbreviations.get(object_type, object_type)
            
            print(f"Debug: Object {i}: type={object_type}, abbreviation={label}, bbox={bbox_coords}")
            
            # Calculate bounding box coordinates based on location information or use provided coordinates
            bbox = calculate_bbox_from_location(location, width, height, i, bbox_coords)
            print(f"Debug: Calculated bbox: {bbox}")
            
            # Draw rectangle with thicker border for better visibility
            draw.rectangle(bbox, outline="red", width=4)
            print(f"Debug: Drew rectangle at {bbox}")
            try:
                # Try to use a font with better size
                font = ImageFont.truetype("arial.ttf", 24)
                print("Debug: Using arial.ttf font")
            except Exception as font_error:
                # Fallback to default font
                font = ImageFont.load_default()
                print(f"Debug: Using default font (error loading arial.ttf: {font_error})")
            
            # Calculate text position (inside the box at the top-left corner)
            text_x, text_y = bbox[0] + 10, bbox[1] + 10
            
            # Draw a semi-transparent background for better text visibility
            text_bbox = draw.textbbox((text_x, text_y), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw background rectangle
            draw.rectangle([text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5], fill="red")
            
            # Draw label with white text
            draw.text((text_x, text_y), label, fill="white", font=font)
            print(f"Debug: Drew label '{label}' at ({text_x}, {text_y})")
        
        # Generate output filename
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        image_ext = os.path.splitext(image_path)[1]
        processed_image_path = os.path.join(output_dir, f"{image_basename}_annotated{image_ext}")
        
        # Save processed image
        image.save(processed_image_path)
        print(f"Processed image saved to: {processed_image_path}")
        
        return processed_image_path
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        # If processing fails, use original image
        return image_path

def calculate_bbox_from_location(location, width, height, index, bbox=None):
    """Calculate bounding box coordinates based on location information or use provided coordinates"""
    # If specific bounding box coordinates are provided, use them directly
    # The API now provides coordinates based on the actual image dimensions
    if bbox and len(bbox) == 4:
        # Use the coordinates directly without scaling
        left = max(0, min(width, int(bbox[0])))
        top = max(0, min(height, int(bbox[1])))
        right = max(0, min(width, int(bbox[2])))
        bottom = max(0, min(height, int(bbox[3])))
        
        print(f"Debug: Using actual coordinates from API: [{left}, {top}, {right}, {bottom}] (actual image size: {width}x{height})")
        
        return [left, top, right, bottom]
    
    # Default grid-based approach as fallback
    grid_size = 3
    box_width = width // grid_size
    box_height = height // grid_size
    
    # Calculate box position based on index
    row = (index - 1) // grid_size
    col = (index - 1) % grid_size
    left = col * box_width
    top = row * box_height
    right = left + box_width
    bottom = top + box_height
    
    # Try to adjust based on location keywords
    if location:
        # Adjust based on horizontal position keywords
        if "左" in location:
            left = 0
            right = width // 3
        elif "右" in location:
            left = 2 * width // 3
            right = width
        elif "中" in location:
            left = width // 4
            right = 3 * width // 4
        
        # Adjust based on vertical position keywords
        if "上" in location:
            top = 0
            bottom = height // 3
        elif "下" in location:
            top = 2 * height // 3
            bottom = height
        elif "中" in location:
            top = height // 4
            bottom = 3 * height // 4
    
    # Ensure coordinates are within bounds
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    
    return [left, top, right, bottom]

def extract_object_info(detection_result):
    """Extract object types, locations, and bounding box coordinates from detection result"""
    object_info = []
    
    print("\nDebug: Extracting object info from detection result")
    
    # Check if the result is in JSON format
    if detection_result.strip().startswith('['):
        try:
            import json
            # Parse JSON
            json_data = json.loads(detection_result)
            print(f"Debug: Detected JSON format, found {len(json_data)} object entries")
            
            for item in json_data:
                object_type = item.get('object_type', 'Unknown Object')
                bbox = item.get('bbox', None)
                position_description = item.get('position_description', '')
                
                if bbox:
                    object_info.append({
                        'type': object_type,
                        'location': position_description,
                        'bbox': bbox
                    })
                    print(f"Debug: Added JSON object entry: type={object_type}, bbox={bbox}")
            
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            # Fall back to text parsing if JSON fails
            object_info = []
    
    # Fall back to text parsing if JSON parsing failed or no results
    if not object_info:
        lines = detection_result.split('\n')
        
        current_object = {}
        
        for line in lines:
            # Debug: Print each line being processed
            print(f"Debug: Processing line: {repr(line)}")
            
            # Check for one-line format: "目标类型, X:XX%-XX%, Y:XX%-XX%, [x1, y1, x2, y2]"
            try:
                import re
                # Match one-line format with object type, location, and bbox
                one_line_pattern = r'(.+?),\s*X:(.+?),\s*Y:(.+?),\s*\[([\d,\s]+)\]'
                match = re.match(one_line_pattern, line.strip())
                
                if match:
                    object_type = match.group(1).strip()
                    x_range = match.group(2).strip()
                    y_range = match.group(3).strip()
                    bbox_str = match.group(4).strip()
                    
                    # Check if object type is supported
                    if object_type in SUPPORTED_OBJECT_TYPES:
                        # Parse bbox coordinates
                        coords = list(map(int, bbox_str.split(',')))
                        if len(coords) == 4:
                            # Create object entry
                            object_entry = {
                                'type': object_type,
                                'location': f"X:{x_range}, Y:{y_range}",
                                'bbox': coords
                            }
                            object_info.append(object_entry)
                            print(f"Debug: Added one-line format object entry: type={object_type}, bbox={coords}")
                            current_object = {}  # Reset current object
                            continue
            except Exception as e:
                print(f"Error parsing one-line format: {e}")
            
            # Check for multi-line format
            # 1. Check if line is an object type
            for object_type in SUPPORTED_OBJECT_TYPES:
                if object_type == line.strip():
                    # Save previous object if it has all required fields
                    if current_object and 'type' in current_object and 'location' in current_object and 'bbox' in current_object:
                        object_info.append(current_object)
                        print(f"Debug: Saved previous object info: {current_object}")
                    
                    # Start new object entry
                    current_object = {'type': object_type}
                    print(f"Debug: Started new object entry for: {object_type}")
                    break
            
            # 2. Check if line is location information
            if current_object and ('X:' in line or 'Y:' in line):
                current_object['location'] = line.strip()
                print(f"Debug: Added location to current object: {line.strip()}")
            
            # 3. Check if line contains bounding box coordinates
            if current_object:
                try:
                    import re
                    # Match coordinates in various formats
                    coord_pattern = r'`?\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]`?'
                    matches = re.findall(coord_pattern, line)
                    
                    if matches:
                        for coord_str in matches:
                            # Remove brackets and backticks
                            clean_coord_str = coord_str.strip('`[]')
                            # Split and convert to integers
                            coords = list(map(int, clean_coord_str.split(',')))
                            if len(coords) == 4:
                                current_object['bbox'] = coords
                                print(f"Debug: Added bbox to current object: {coords}")
                                
                                # If this object has all required fields, save it
                                if 'type' in current_object and 'location' in current_object and 'bbox' in current_object:
                                    object_info.append(current_object)
                                    print(f"Debug: Saved complete object info: {current_object}")
                                    current_object = {}  # Reset for next object
                except Exception as e:
                    print(f"Error extracting coordinates: {e}")
        
        # Save any remaining object entry
        if current_object and 'type' in current_object and 'location' in current_object and 'bbox' in current_object:
            object_info.append(current_object)
            print(f"Debug: Saved final object info: {current_object}")
    
    # Filter out object entries without bbox coordinates
    filtered_object_info = []
    for obj in object_info:
        if 'bbox' in obj and obj['bbox']:
            filtered_object_info.append(obj)
            print(f"Debug: Keeping object with bbox: {obj['type']}")
        else:
            print(f"Debug: Skipping object without bbox: {obj.get('type', 'Unknown')}")
    
    print(f"Debug: Final object info list: {filtered_object_info}")
    
    return filtered_object_info

def generate_report(detection_result, image_path, analyzed_image_path, output_dir, format="md"):
    """Generate detection report for a single image"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process image (draw bounding boxes) using the same image path as analysis
    processed_image_path = draw_object_boxes(analyzed_image_path, detection_result, output_dir)
    
    # Generate report filename from image basename
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Generate report content
    if format == "md":
        report_content = generate_markdown_report(detection_result, image_path, processed_image_path)
        output_file = os.path.join(output_dir, f"{image_basename}_report.md")
    else:
        report_content = generate_text_report(detection_result, image_path)
        output_file = os.path.join(output_dir, f"{image_basename}_report.txt")
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\nDetection report generated: {output_file}")
    return output_file

def generate_markdown_report(detection_result, image_path, processed_image_path):
    """Generate Markdown format report with processed image"""
    import datetime
    
    report = "# 目标检测报告\n\n"
    report += f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"分析图像: {os.path.basename(image_path)}\n\n"
    report += f"图像路径: {image_path}\n\n"
    
    # Add processed image to the report
    report += "## 标注后图像\n\n"
    report += f"![标注后图像]({os.path.basename(processed_image_path)})\n\n"
    
    report += "## 检测结果\n\n"
    report += f"{detection_result}\n\n"
    
    report += "## 检测说明\n\n"
    report += "本报告检测以下目标类型：\n\n"
    for object_type in SUPPORTED_OBJECT_TYPES:
        report += f"- {object_type}\n"
    
    return report

def generate_text_report(detection_result, image_path):
    """Generate text format report"""
    import datetime
    
    report = "=" * 80 + "\n"
    report += "目标检测报告\n"
    report += "=" * 80 + "\n"
    report += f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"分析图像: {os.path.basename(image_path)}\n"
    report += f"图像路径: {image_path}\n"
    report += "=" * 80 + "\n\n"
    
    report += "检测结果:\n"
    report += "-" * 60 + "\n"
    report += f"{detection_result}\n\n"
    
    report += "检测说明:\n"
    report += "-" * 60 + "\n"
    report += "本报告检测以下目标类型：\n"
    for object_type in SUPPORTED_OBJECT_TYPES:
        report += f"- {object_type}\n"
    report += "=" * 80 + "\n"
    
    return report

def process_single_image(image_path, args, output_dir):
    """Process a single image and generate report"""
    # Check if report already exists
    if args.skip_existing:
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        report_file = os.path.join(output_dir, f"{image_basename}_report.{args.format}")
        if os.path.exists(report_file):
            print(f"\nSkipping image: {image_path}")
            print(f"Report already exists: {report_file}")
            return True
    
    print(f"\nProcessing image: {image_path}")
    print("-" * 80)
    
    # Analyze the image
    detection_result, analyzed_image_path = analyze_image_with_siliconflow(image_path, args.api_key, args.base_url)
    
    # Print detection result
    print("\n目标检测结果:")
    print("-" * 60)
    try:
        print(detection_result)
    except UnicodeEncodeError:
        # Handle encoding errors
        print(detection_result.encode('gbk', errors='replace').decode('gbk'))
    print("-" * 60)
    
    # Generate report
    generate_report(detection_result, image_path, analyzed_image_path, output_dir, args.format)
    
    return True

def main():
    """Main function"""
    args = parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.dir):
        print(f"错误: 目录不存在: {args.dir}")
        return
    
    # Check if specific file is provided
    if args.file:
        if os.path.exists(args.file):
            selected_images = [args.file]
            print(f"Processing specific file: {args.file}")
        else:
            print(f"错误: 文件不存在: {args.file}")
            return
    else:
        # Get all image files in the directory
        image_files = get_image_files(args.dir)
        if not image_files:
            print(f"错误: 目录中未找到支持的图像文件: {args.dir}")
            return
        
        # Select images: process all if count is 0, otherwise random select
        if args.count == 0:
            selected_images = image_files
            print(f"Processing all {len(selected_images)} images")
        else:
            selected_images = random_select_images(image_files, args.count)
    
    # Print header
    print("=" * 80)
    print("目标检测 (批处理模式)")
    print("=" * 80)
    if args.file:
        print(f"分析文件: {args.file}")
    else:
        print(f"分析目录: {args.dir}")
    print(f"处理图像数: {len(selected_images)}")
    print(f"输出目录: {args.output}")
    print("=" * 80)
    
    # 支持的目标类型
    print("\n检测以下目标类型:")
    for i, object_type in enumerate(SUPPORTED_OBJECT_TYPES, 1):
        print(f"{i}. {object_type}")
    
    # Process selected images
    success_count = 0
    for image_path in selected_images:
        try:
            if process_single_image(image_path, args, args.output):
                success_count += 1
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"批处理完成! 成功处理 {success_count}/{len(selected_images)} 张图像")
    print("=" * 80)

if __name__ == "__main__":
    main()
