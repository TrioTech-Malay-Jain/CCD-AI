"""
File upload and management API endpoints
"""
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from models.schemas import FileInfo, FileUploadRequest
from services.file_service import file_service
from services.embedding_service import embedding_service

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload", response_model=FileInfo)
async def upload_file(
    background_tasks: BackgroundTasks,
    company_id: str = Form(..., description="Company ID for file isolation"),
    file: UploadFile = File(..., description="File to upload")
):
    """Upload a file for a specific company"""
    
    try:
        # Save file
        file_info = await file_service.save_uploaded_file(company_id, file)
        
        # Add background task to process document
        background_tasks.add_task(
            embedding_service.add_document_to_company, 
            company_id, 
            file_info.file_id
        )
        
        return file_info
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.get("/list", response_model=List[FileInfo])
async def list_files(company_id: str):
    """List all files for a specific company"""
    
    try:
        files = file_service.list_company_files(company_id)
        return files
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.get("/{file_id}", response_model=FileInfo)
async def get_file_info(company_id: str, file_id: str):
    """Get information about a specific file"""
    
    file_info = file_service.get_file_info(company_id, file_id)
    
    if not file_info:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found for company {company_id}")
    
    return file_info


@router.delete("/{file_id}")
async def delete_file(company_id: str, file_id: str):
    """Delete a specific file"""
    
    try:
        # Delete from file system
        success = file_service.delete_file(company_id, file_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        
        # Delete from vector store
        from db.chroma_manager import chroma_manager
        chroma_manager.delete_documents_from_company(company_id, [file_id])
        
        return {"message": f"File {file_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@router.get("/stats/{company_id}")
async def get_file_stats(company_id: str):
    """Get file statistics for a company"""
    
    try:
        stats = file_service.get_company_file_stats(company_id)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get file stats: {str(e)}")
