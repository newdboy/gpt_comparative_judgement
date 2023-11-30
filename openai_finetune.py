# FINETUNE
import os
import openai
import dotenv

# api key
dotenv.load_dotenv(dotenv.find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

def ft_upload_file(file_dir):
    uploaded = openai.File.create(
        file = open(file_dir, "rb"),
        purpose = 'fine-tune',
        user_provided_filename = file_dir.split('/')[-1]
    )
    return uploaded


# # delete all the uploaded files
# uploaded_file_list = openai.File.list(limit=10)
# for k in range(len(uploaded_file_list)):
#     file_id = uploaded_file_list['data'][k]['id']
#     print(file_id)
#     openai.File.delete(sid=file_id)
#
# # confirm files
# openai.File.list()



# To check result files EXECUTE BELOWS IN TERMINAL
#Retrieve file content
# curl https://api.openai.com/v1/files/{file_id}/content \
#   -H "Authorization: Bearer $OPENAI_API_KEY" > file.jsonl

# curl https://api.openai.com/v1/files/file-F0cWez8qHfKpH4NVxZxK3aWi/content \
#   -H "Authorization: Bearer OPENAI_API_KEY" > finetune_resultfile_091013.jsonl