from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
import time
import urllib.parse

# Ask the user for the item to search on AliExpress
search_query = input("Enter the item you want to search for on AliExpress: ")
encoded_query = urllib.parse.quote(search_query)
search_url = f"https://www.aliexpress.com/wholesale?SearchText={encoded_query}"

# Headers for requests
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'en-GB,en;q=0.8',
}

# Fetch the search results page using requests
response = requests.get(search_url, headers=headers)
html_text = response.text
soup = BeautifulSoup(html_text, 'lxml')

# Find multiple item listings (e.g., first 5 items)
items = soup.find_all('div', class_='multi--modalContext--1Hxqhwi', limit=5)

# Setup Selenium WebDriver (Only start it once for efficiency)
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Process each item
for index, item in enumerate(items, start=1):
    # Extract item name
    item_name_tag = item.find('a', class_='multi--container--1UZxxHY cards--card--3PJxwBm search-card-item')
    item_name = item_name_tag.text.strip() if item_name_tag else "N/A"

    # Extract company name
    shop_name_tag = item.find('span', class_='cards--store--3GyJcot')
    shop_name = shop_name_tag.text.strip() if shop_name_tag else "N/A"
    shop_name = ' '.join(shop_name.split())

    # Extract price
    price_tag = item.find('div', class_='multi--price-sale--U-S0jtj')
    price = price_tag.text.strip() if price_tag else "N/A"

    # Extract item URL
    item_link_tag = item.find('a', class_='multi--container--1UZxxHY cards--card--3PJxwBm search-card-item')
    if item_link_tag and 'href' in item_link_tag.attrs:
        item_url = item_link_tag['href']
        if item_url.startswith('//'):
            item_url = 'https:' + item_url  # Fix relative URLs
    else:
        item_url = "N/A"

    # Extract stock information using Selenium
    if item_url != "N/A":
        try:
            driver.get(item_url)
            time.sleep(5)  # Wait for JavaScript to load content

            # Parse item page with BeautifulSoup
            item_soup = BeautifulSoup(driver.page_source, 'lxml')

            # Extract stock information
            stock_div = item_soup.find('div', class_='quantity--info--jnoo_pD')
            stock_text = stock_div.text.strip() if stock_div else "Stock info not found."

        except Exception as e:
            stock_text = f"Error fetching stock: {e}"

    else:
        stock_text = "N/A"

    # Print formatted output for each item
    print(f'''
    Company Name: {shop_name}
    Item URL: {item_url}
    Item Name: {item_name}
    Price: {price}
    Stock Available: {stock_text}
    ''')

# Close the Selenium WebDriver after processing all items
driver.quit()
