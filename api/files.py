"""
File upload and management API endpoints
"""
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from models.schemas import FileInfo, FileUploadRequest, BuildStatus
from services.file_service import file_service
from services.embedding_service import embedding_service

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload", response_model=FileInfo)
async def upload_file(
    background_tasks: BackgroundTasks,
    company_id: str = Form(..., description="Company ID for file isolation"),
    file: UploadFile = File(..., description="File to upload")
):
    """Upload a file for a specific company and automatically build vector database"""
    
    try:
        # Save file
        file_info = await file_service.save_uploaded_file(company_id, file)
        
        # Check if company directory exists
        company_dir = file_service.get_company_directory(company_id)
        if not company_dir.exists():
            raise HTTPException(status_code=404, detail=f"Company {company_id} not found")
        
        # Check if already building
        current_status = embedding_service.get_build_status(company_id)
        if current_status.status != "building":
            # Start build process in background for the entire company
            background_tasks.add_task(
                embedding_service.process_company_documents, 
                company_id
            )
        
        # Also create file-specific collection for targeted querying
        background_tasks.add_task(
            embedding_service.create_file_specific_collection,
            company_id,
            file_info.file_id
        )
        
        # Also add the individual document processing as fallback
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


@router.get("/build-status/{company_id}", response_model=BuildStatus)
async def get_build_status(company_id: str):
    """Get build status for a company's vector database"""
    
    return embedding_service.get_build_status(company_id)
