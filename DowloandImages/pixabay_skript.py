import pixabay_python as pxb
import os
import time

def main(output_dir: str, target_count: int = 300):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize the Pixabay client with your API key
        client = pxb.PixabayClient(apiKey="51731683-e8f90bd3ff054e7513f2316ac")
        
        total_downloaded = 0
        page = 1
        per_page = 200  # Maximum allowed by Pixabay API per request
        max_pages = (target_count + per_page - 1) // per_page  # Calculate needed pages
        
        print(f"🎯 Target: {target_count} images")
        print(f"📄 Will search up to {max_pages} pages with {per_page} results per page")
        
        while total_downloaded < target_count and page <= max_pages:
            print(f"\n🔍 Searching page {page}...")
            
            try:
                # Search with pagination
                searchResult = client.searchImage(
                    q="Statue Of Liberty",  # Replace with your working search term
                    imageType=pxb.ImageType.PHOTO,
                    perPage=per_page,
                    page=page,
                    minWidth=640,
                    minHeight=480
                )
                
                page_count = 0
                
                # Download images from current page
                for hit in searchResult.hits:
                    if total_downloaded >= target_count:
                        break
                    
                    try:
                        # Choose the best available URL
                        if hasattr(hit, 'largeImageURL'):
                            download_url = hit.largeImageURL
                        elif hasattr(hit, 'webformatURL'):
                            download_url = hit.webformatURL
                        else:
                            print(f"⚠️  Skipping image - no usable URL")
                            continue
                        
                        # Download the image
                        pxb.download(url=download_url, outputDir=output_dir)
                        total_downloaded += 1
                        page_count += 1
                        
                        # Progress update every 10 images
                        if total_downloaded % 10 == 0:
                            print(f"📥 Downloaded {total_downloaded}/{target_count} images...")
                        
                        # Small delay to be respectful to the API
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"❌ Error downloading image: {str(e)}")
                        continue
                
                print(f"✅ Page {page} complete: {page_count} images downloaded")
                page += 1
                
                # Longer delay between pages to avoid rate limiting
                if page <= max_pages and total_downloaded < target_count:
                    print("⏳ Waiting 2 seconds before next page...")
                    time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error on page {page}: {str(e)}")
                break
        
        print(f"\n🎉 Download complete! Got {total_downloaded} images in '{output_dir}'")
        
        if total_downloaded < target_count:
            print(f"⚠️  Note: Only found {total_downloaded} images (less than target of {target_count})")
        
    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")

def download_with_multiple_search_terms(output_dir: str, target_count: int = 300):
    """Alternative approach: Use multiple related search terms to get more variety"""
    
    # List of related search terms - modify these based on what you're looking for
    search_terms = [
        "your_main_term",      # Replace with your main search term
        "related_term_1",      # Add related terms
        "related_term_2",      # to get more variety
        "related_term_3",
        # Add more terms as needed
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    client = pxb.PixabayClient(apiKey="51731683-e8f90bd3ff054e7513f2316ac")
    
    total_downloaded = 0
    images_per_term = target_count // len(search_terms)
    
    print(f"🎯 Target: {target_count} images using {len(search_terms)} search terms")
    print(f"📊 ~{images_per_term} images per search term")
    
    for i, term in enumerate(search_terms):
        if total_downloaded >= target_count:
            break
            
        print(f"\n🔍 Searching for '{term}' ({i+1}/{len(search_terms)})...")
        
        try:
            searchResult = client.searchImage(
                q=term,
                imageType=pxb.ImageType.PHOTO,
                perPage=min(200, target_count - total_downloaded),
                minWidth=640,
                minHeight=480,
                safesearch="true"
            )
            
            term_count = 0
            for hit in searchResult.hits:
                if total_downloaded >= target_count:
                    break
                if term_count >= images_per_term and i < len(search_terms) - 1:
                    break  # Move to next term (except for last term)
                
                try:
                    download_url = hit.largeImageURL if hasattr(hit, 'largeImageURL') else hit.webformatURL
                    pxb.download(url=download_url, outputDir=output_dir)
                    total_downloaded += 1
                    term_count += 1
                    
                    if total_downloaded % 25 == 0:
                        print(f"📥 Downloaded {total_downloaded}/{target_count} images...")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"❌ Error downloading: {str(e)}")
                    continue
            
            print(f"✅ '{term}': {term_count} images downloaded")
            time.sleep(1)  # Pause between search terms
            
        except Exception as e:
            print(f"❌ Error searching '{term}': {str(e)}")
            continue
    
    print(f"\n🎉 All downloads complete! Got {total_downloaded} images in '{output_dir}'")

if __name__ == "__main__":
    output_dir = "downloaded_images_300"
    
    # Choose one of these approaches:
    
    # Approach 1: Single search term with pagination
    print("🚀 Starting single-term download with pagination...")
    main(output_dir, target_count=300)
