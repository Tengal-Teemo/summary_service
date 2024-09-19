# API Documentation
## Summarize Documents
```http request
POST tools.zagmo.net/summarize
```
### Requires
Requires a `multipart/form-data` body, containing:
 - `earlier`: list of first versions of files
 - `later`: list of second versions of files
 - `roles`: list of roles to filter for, e.g. CEO, Software engineer

### Returns
 - A stringified json object

```json
{"body": 
    {
      "laterFile1": {
        "role1": "Summary of changes for role 1 for doc 1",
        "role2": "Summary of changes for role 2 for doc 1"
      },
      "laterFile2": {
        "role1": "Summary of changes for role 1 for doc 2",
        "role2": "Summary of changes for role 2 for doc 2"
      }
    }
}
```