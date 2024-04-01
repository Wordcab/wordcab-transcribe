# Copyright 2024 The Wordcab Team. All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Wordcab/wordcab-transcribe/blob/main/LICENSE
#
# Except as expressly provided otherwise herein, and to the fullest
# extent permitted by law, Licensor provides the Software (and each
# Contributor provides its Contributions) AS IS, and Licensor
# disclaims all warranties or guarantees of any kind, express or
# implied, whether arising under any law or from any usage in trade,
# or otherwise including but not limited to the implied warranties
# of merchantability, non-infringement, quiet enjoyment, fitness
# for a particular purpose, or otherwise.
#
# See the License for the specific language governing permissions
# and limitations under the License.
"""Authentication dependency for production."""

from datetime import datetime, timedelta
from typing import Union

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from loguru import logger

from wordcab_transcribe.config import settings
from wordcab_transcribe.models import Token, TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/auth")

credentials_exception = HTTPException(
    status_code=http_status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

router = APIRouter()


def _get_username() -> str:
    return settings.username


def create_access_token(
    data: dict, expires_delta: Union[timedelta, None] = None
) -> str:
    """
    Create access token for user authentication.

    Args:
        data (dict): Data to be encoded in the token.
        expires_delta (Union[timedelta, None], optional): Expiration time of the token. Defaults to None.

    Returns:
        str: Access token.
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode, settings.openssl_key, algorithm=settings.openssl_algorithm
    )


async def get_current_user(
    token: str = Depends(oauth2_scheme),  # noqa: B008
    credentials: str = Depends(_get_username),  # noqa: B008
) -> str:
    """
    Get current user dependency function for authentication. Not meant to be used with Cortex endpoint.

    Args:
        token (str): Access token. Depends(oauth2_scheme).
        credentials (str): Username. Depends(_get_username).

    Raises:
        credentials_exception: If the credentials are not valid.

    Returns:
        str: Username.
    """
    try:
        payload = jwt.decode(
            token, settings.openssl_key, algorithms=[settings.openssl_algorithm]
        )
        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception

        token_data = TokenData(username=username)

    except JWTError as e:
        raise credentials_exception from e

    if token_data.username != credentials:
        raise credentials_exception

    return username


async def authenticate_user(username: str, password: str) -> dict:
    """
    Authenticate user dependency function for authentication.

    Args:
        username (str): Username.
        password (str): Password.

    Raises:
        HTTPException: If the username or password are incorrect.

    Returns:
        dict: Access token and token type.
    """
    if username != settings.username or password != settings.password:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=1440)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.post(
    f"{settings.api_prefix}/auth",
    response_model=Token,
    status_code=http_status.HTTP_200_OK,
)
async def authentication(
    form_data: OAuth2PasswordRequestForm = Depends(),  # noqa: B008
):
    """Authentication endpoint for the API."""
    user = await authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.debug(f"Authenticating user {form_data.username}")
    return user
