from parser_agents import extract_text_from_pdf, build_chunks, CHUNK_SIZE, OVERLAP
from config import DOC_PATHS
 
print(f"CHUNK_SIZE = {CHUNK_SIZE}")
print(f"OVERLAP    = {OVERLAP}")
print()
 
text = extract_text_from_pdf(DOC_PATHS["annual_report"])
print(f"Total document length: {len(text):,} characters")
 
chunks = build_chunks(text)
print(f"Document split into {len(chunks)} chunk(s).")
print()
 
# Show size of each chunk to sanity-check the split
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i + 1}: {len(chunk):,} chars")